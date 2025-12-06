/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "ThreadedDecoder.hpp"
#include "PyNvVideoCodecUtils.hpp"

ThreadedDecoder::ThreadedDecoder(const std::string& encSource,
            uint32_t bufferSize,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudaStream,
            bool useDeviceMemory,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool needScannedStreamMetadata,
            uint32_t decoderCacheSize,
            OutputColorType outputColorType,
            uint32_t startFrame) : mDecodedFrames(bufferSize), mStartFrame(startFrame)
{
    mDecoderCommon.reset(new DecoderCommon(encSource, gpuId, cudaContext, cudaStream, useDeviceMemory, maxWidth,
                        maxHeight, needScannedStreamMetadata, decoderCacheSize, outputColorType));
}

ThreadedDecoder::~ThreadedDecoder()
{
    if (!endCalled)
    {
        End();
    }
}

void ThreadedDecoder::Initialize()
{
    endCalled = false;
    mPrevBatchSize = 0;
    mDecodeStopFlag.store(false);

    // Get total frame count for bounds checking
    StreamMetadata metadata = mDecoderCommon->GetStreamMetadata();
    uint32_t totalFrames = metadata.numFrames;
    
    // Check if startFrame is within bounds
    bool validStartFrame = (mStartFrame < totalFrames);
    if (!validStartFrame) {
        LOG(ERROR) << "startFrame (" << mStartFrame << ") is >= total frames (" << totalFrames << ")";
        LOG(ERROR) << "No frames will be decoded. Decoder will return empty results.";
        mNearestKeyFrameIdx = 0;
    } else {
        // Compute nearest key frame index for startFrame
        if (mStartFrame > 0) {
            SeekUtils* seekUtils = mDecoderCommon->GetPtrToSeekUtils();
            mNearestKeyFrameIdx = seekUtils->GetKeyNearestKeyFrameIndexForTarget(mDecoderCommon->GetDemuxer()->GetVideoStream(), mStartFrame);
            LOG(DEBUG) << "Computed nearest key frame index: " << mNearestKeyFrameIdx << " for startFrame: " << mStartFrame;
        } else {
            mNearestKeyFrameIdx = 0;
        }
    }

    // Create thread for decoding - pass validity flag and computed key frame index
    mDecoderThread = NvThread(std::thread(RunDecoder<DecodedFrame>, mDecoderCommon.get(), mDecoderCommon->GetDemuxer(), mDecoderCommon->GetDecoder(),
                    std::ref(mDecodedFrames), std::ref(mDecodeStopFlag), static_cast<int>(mStartFrame), validStartFrame, mNearestKeyFrameIdx));
}

template <typename T>
static void RunDecoder(DecoderCommon* decoderCommon, FFmpegDemuxer* demuxer, NvDecoder* decoder, SPSCBuffer<T>& decodedFrames,
            std::atomic<bool>& decodeStopFlag, int startFrame, bool validStartFrame, uint32_t nearestKeyFrameIdx)
{
    // If startFrame is invalid, immediately signal done and return
    if (!validStartFrame) {
        LOG(ERROR) << "Invalid startFrame, skipping decode and signaling done";
        decodedFrames.PushDone();
        return;
    }
    
    int nVideoBytes = 0, nFrameReturned = 0;
    uint8_t* pVideo = NULL;
    int64_t pts = 0;
    int64_t dts = 0;
    uint64_t duration = 0;
    uint64_t pos = 0;
    bool keyFrame = false;
    
    // Seek to nearest keyframe before startFrame if needed
    if (startFrame > 0) {
        demuxer->Seek(startFrame);
        LOG(DEBUG) << "Using pre-computed key frame index: " << nearestKeyFrameIdx;
        LOG(DEBUG) << "Seeking to frame: " << startFrame;
    }
    
    // Frame counter starting from the key frame index
    int currentFrameIndex = nearestKeyFrameIdx;
    
    // Main decoding loop - decode every frame and decide whether to keep or discard
    do {
        demuxer->Demux(&pVideo, &nVideoBytes, pts, dts, duration, pos, keyFrame);
        nFrameReturned = decoder->Decode(pVideo, nVideoBytes, 0, pts);
        
        for (int i = 0; (i < nFrameReturned) && (!decodeStopFlag.load()); i++) {
            int64_t timestamp = 0;
            SEI_MESSAGE seims;
            CUevent event = nullptr;
            CUdeviceptr dpFrame = (CUdeviceptr)decoder->GetLockedFrame(&timestamp, &seims, &event);
            if (dpFrame) {
                // Use frame counter based on key frame index instead of dts_to_frame_number
                int frameNumber = currentFrameIndex;
                currentFrameIndex++; // Increment frame counter for next frame
                
                // Check if this frame should be kept or discarded
                if (frameNumber >= startFrame) {
                    LOG(DEBUG) << "Keeping frame: " << frameNumber;
                    DecodedFrame decodedFrame = GetCAIMemoryViewAndDLPack(decoder, std::make_tuple(dpFrame, timestamp, seims, event));
                    decodedFrames.PushEntry(decodedFrame);
                } else {
                    LOG(DEBUG) << "Discarding frame: " << frameNumber;
                    // Just unlock the frame without keeping it
                    decoder->UnlockFrame((uint8_t*)dpFrame);
                }
            }
        }
    } while (nVideoBytes && !decodeStopFlag.load());
    
    decodedFrames.PushDone();
    
    // Send empty packet to decoder to simulate decode complete
    if (nVideoBytes != 0) {
        PacketData emptyPacket = PacketData();
        decoder->Decode((uint8_t*)emptyPacket.bsl_data, emptyPacket.bsl);
    }
}

void ThreadedDecoder::End()
{
    // Will stop any further decode
    mDecodeStopFlag.store(true);
    // This is not a clean solution. The PushEntry call tries to push a "locked frame"
    // but goes to sleep because there is no space in the buffer. We will need to wake that
    // thread to shut it down. So we call GetBatchFrames with size 1. This will wake the thread
    // and push the locked frame. This and abvoe effectively stops the Push thread.
    GetBatchFrames(1);
    mDecoderThread.join();
    
    // Drain the buffer so as to unlock the frames
    GetBatchFrames(0);
    mDecoderCommon->UnlockLockedFrames(mPrevBatchSize);

    // reset state
    mPrevBatchSize = 0;
    mDecodeStopFlag.store(false);
    mDecodedFrames.Clear();
    endCalled = true;
}

std::vector<DecodedFrame> ThreadedDecoder::GetBatchFrames(size_t batchSize)
{
    // unlock previously locked frames if any
    py::gil_scoped_release release;
    mDecoderCommon->UnlockLockedFrames(mPrevBatchSize);
    auto frames = mDecodedFrames.PopEntries(batchSize);
    mPrevBatchSize = frames.size();
    py::gil_scoped_acquire acquire;
    return frames;
}

ScannedStreamMetadata ThreadedDecoder::GetScannedStreamMetadata()
{
    return mDecoderCommon->GetScannedStreamMetadata();
}

StreamMetadata ThreadedDecoder::GetStreamMetadata()
{
    return mDecoderCommon->GetStreamMetadata();
}

void ThreadedDecoder::ReconfigureDecoder(std::string newSource)
{
    // Gracefully shutdown the current decoder thread.
    End();
    mDecoderCommon->ReconfigureDecoder(newSource);
    // Intialize the decoder again
    Initialize();
}