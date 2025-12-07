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

#pragma once

#include "NvDemuxer.hpp"
#include "FFmpegMuxer.h"

/**
 * @brief Thin C++ wrapper around FFmpegMuxer that operates on PacketData
 *        produced by FFmpegDemuxer / NvDemuxer.
 */
class NvMuxer {
private:
    std::unique_ptr<FFmpegMuxer> muxer;
    int videoStreamIndex = -1;
    AVFormatContext* inputFmtc = nullptr;
    bool ownsInputFmtc = false;

public:
    /**
     * @brief Construct a muxer for the given output file using stream
     *        metadata from the provided FFmpegDemuxer instance.
     *
     * The constructor inspects the input container, derives codec, width,
     * height and codec extradata from the video stream, and creates a
     * corresponding FFmpegMuxer instance.
     */
    NvMuxer(const std::string& outputFilePath, FFmpegDemuxer* demuxer);

    /**
     * @brief Construct a muxer for the given output file using explicit
     *        stream properties (codec, resolution, frame rate).
     *
     * This overload is intended for workflows that encode from raw frames
     * without an existing container (e.g., using PyNvEncoder). A synthetic
     * AVFormatContext is created internally to drive FFmpegMuxer.
     */
    NvMuxer(const std::string& outputFilePath,
            AVCodecID codecId,
            int width,
            int height,
            AVRational fps);

    ~NvMuxer();

    /**
     * @brief Mux a single PacketData (compressed video packet) into the
     *        target container.
     *
     * @param packet PacketData produced by the associated demuxer.
     * @return true on success, false otherwise.
     */
    bool Mux(const PacketData& packet);

    /**
     * @brief Mux a raw encoded bitstream packet into the target container.
     *
     * This is a convenience wrapper around FFmpegMuxer for callers that
     * already have compressed bitstream bytes and explicit timing.
     */
    bool MuxRaw(const uint8_t* data,
                unsigned int size,
                int64_t pts,
                int64_t dts,
                int64_t duration,
                bool isKeyFrame);
};
