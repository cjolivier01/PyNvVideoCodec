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

#include "NvMuxer.hpp"

NvMuxer::NvMuxer(const std::string& outputFilePath, FFmpegDemuxer* demuxer)
{
    if (!demuxer) {
        PYNVVC_THROW_ERROR("NvMuxer constructor failed: demuxer is null", CUDA_ERROR_NOT_SUPPORTED);
    }

    MEDIA_FORMAT mediaFormat = GetMediaFormat(outputFilePath);

    size_t extradataSize = 0;
    const uint8_t* extradataConst = demuxer->GetVideoExtradata(extradataSize);
    uint8_t* extradata = nullptr;
    if (extradataConst != nullptr && extradataSize > 0) {
        extradata = const_cast<uint8_t*>(extradataConst);
    }

    AVCodecID codecId = demuxer->GetVideoCodec();
    int width = demuxer->GetWidth();
    int height = demuxer->GetHeight();
    videoStreamIndex = demuxer->GetVideoStreamId();

    inputFmtc = demuxer->GetAVFormatContext();
    ownsInputFmtc = false;

    muxer.reset(new FFmpegMuxer(
        outputFilePath.c_str(),
        mediaFormat,
        inputFmtc,
        codecId,
        width,
        height,
        extradata,
        extradataSize));
}

NvMuxer::NvMuxer(const std::string& outputFilePath,
                 AVCodecID codecId,
                 int width,
                 int height,
                 AVRational fps)
{
    if (width <= 0 || height <= 0) {
        PYNVVC_THROW_ERROR("NvMuxer constructor failed: invalid width/height", CUDA_ERROR_NOT_SUPPORTED);
    }

    if (fps.num <= 0 || fps.den <= 0) {
        // Default to 30 fps if invalid.
        fps.num = 30;
        fps.den = 1;
    }

    MEDIA_FORMAT mediaFormat = GetMediaFormat(outputFilePath);

    inputFmtc = avformat_alloc_context();
    if (!inputFmtc) {
        PYNVVC_THROW_ERROR("NvMuxer constructor failed: avformat_alloc_context() returned null", CUDA_ERROR_NOT_SUPPORTED);
    }

    AVStream* stream = avformat_new_stream(inputFmtc, nullptr);
    if (!stream) {
        avformat_free_context(inputFmtc);
        inputFmtc = nullptr;
        PYNVVC_THROW_ERROR("NvMuxer constructor failed: avformat_new_stream() returned null", CUDA_ERROR_NOT_SUPPORTED);
    }

    stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    stream->codecpar->codec_id = codecId;
    stream->codecpar->width = width;
    stream->codecpar->height = height;

    // Use fps as the average frame rate and derive a matching time_base.
    stream->avg_frame_rate = fps;
    stream->time_base.num = fps.den;
    stream->time_base.den = fps.num;

    videoStreamIndex = stream->index;
    ownsInputFmtc = true;

    muxer.reset(new FFmpegMuxer(
        outputFilePath.c_str(),
        mediaFormat,
        inputFmtc,
        codecId,
        width,
        height,
        nullptr,
        0));
}

NvMuxer::~NvMuxer()
{
    muxer.reset();
    if (ownsInputFmtc && inputFmtc) {
        avformat_free_context(inputFmtc);
        inputFmtc = nullptr;
    }
}

bool NvMuxer::Mux(const PacketData& packet)
{
    if (!muxer) {
        return false;
    }

    if (packet.bsl == 0 || packet.bsl_data == 0) {
        return false;
    }

    auto* data = reinterpret_cast<uint8_t*>(packet.bsl_data);
    unsigned int size = static_cast<unsigned int>(packet.bsl);

    int64_t pts = packet.pts;
    int64_t dts = packet.dts;
    int64_t duration = static_cast<int64_t>(packet.duration);
    int isKeyFrame = packet.key ? 1 : 0;

    // For simple remux/transcode flows, we do not use the numb parameter,
    // so pass 0 here.
    return muxer->Mux(data, size, pts, dts, duration, videoStreamIndex, isKeyFrame, 0);
}

bool NvMuxer::MuxRaw(const uint8_t* data,
                     unsigned int size,
                     int64_t pts,
                     int64_t dts,
                     int64_t duration,
                     bool isKeyFrame)
{
    if (!muxer || !data || size == 0) {
        return false;
    }

    int key = isKeyFrame ? 1 : 0;
    return muxer->Mux(const_cast<uint8_t*>(data),
                      size,
                      pts,
                      dts,
                      duration,
                      videoStreamIndex,
                      key,
                      0);
}
