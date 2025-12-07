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

#include "PyNvDemuxer.hpp"
#include "NvMuxer.hpp"

namespace py = pybind11;

class PyNvMuxer {
protected:
    std::unique_ptr<NvMuxer> muxer;
    std::shared_ptr<PyNvDemuxer> demuxer;
    int64_t nextPts = 0;
    bool firstPacket = true;

public:
    PyNvMuxer(const std::string& outputFilePath, std::shared_ptr<PyNvDemuxer> demuxerIn)
        : demuxer(std::move(demuxerIn))
    {
        try {
            if (!demuxer) {
                PYNVVC_THROW_ERROR("PyNvMuxer constructor failed: demuxer is null", CUDA_ERROR_NOT_SUPPORTED);
            }

            NvDemuxer* nvDemuxer = demuxer->GetNvDemuxer();
            if (!nvDemuxer) {
                PYNVVC_THROW_ERROR("PyNvMuxer constructor failed: NvDemuxer is null", CUDA_ERROR_NOT_SUPPORTED);
            }

            FFmpegDemuxer* ffmpegDemuxer = nvDemuxer->GetFFmpegDemuxer();
            if (!ffmpegDemuxer) {
                PYNVVC_THROW_ERROR("PyNvMuxer constructor failed: FFmpegDemuxer is null", CUDA_ERROR_NOT_SUPPORTED);
            }

            muxer.reset(new NvMuxer(outputFilePath, ffmpegDemuxer));
        }
        catch (const PyNvVCException<PyNvVCGenericError>&) {
            throw;
        }
        catch (const PyNvVCException<PyNvVCUnsupported>&) {
            throw;
        }
        catch (const std::exception& e) {
            PYNVVC_THROW_ERROR(std::string("PyNvMuxer constructor failed: ") + e.what(), CUDA_ERROR_UNKNOWN);
        }
        catch (...) {
            PYNVVC_THROW_ERROR("PyNvMuxer constructor failed with unknown error", CUDA_ERROR_UNKNOWN);
        }
    }

    PyNvMuxer(const std::string& outputFilePath,
              const std::string& codec,
              int width,
              int height,
              double fps)
    {
        try {
            if (width <= 0 || height <= 0) {
                PYNVVC_THROW_ERROR("PyNvMuxer constructor failed: invalid width/height", CUDA_ERROR_NOT_SUPPORTED);
            }

            AVCodecID codecId;
            std::string codecLower = codec;
            std::transform(codecLower.begin(), codecLower.end(), codecLower.begin(), ::tolower);
            if (codecLower == "h264" || codecLower == "avc1") {
                codecId = AV_CODEC_ID_H264;
            }
            else if (codecLower == "hevc" || codecLower == "h265") {
                codecId = AV_CODEC_ID_HEVC;
            }
            else if (codecLower == "av1") {
                codecId = AV_CODEC_ID_AV1;
            }
            else {
                PYNVVC_THROW_ERROR_UNSUPPORTED("Unsupported codec for PyNvMuxer", CUDA_ERROR_NOT_SUPPORTED);
            }

            if (fps <= 0.0) {
                fps = 30.0;
            }

            AVRational fpsRational;
            fpsRational.num = static_cast<int>(fps * 1000.0 + 0.5);
            fpsRational.den = 1000;

            muxer.reset(new NvMuxer(outputFilePath, codecId, width, height, fpsRational));
        }
        catch (const PyNvVCException<PyNvVCGenericError>&) {
            throw;
        }
        catch (const PyNvVCException<PyNvVCUnsupported>&) {
            throw;
        }
        catch (const std::exception& e) {
            PYNVVC_THROW_ERROR(std::string("PyNvMuxer constructor failed: ") + e.what(), CUDA_ERROR_UNKNOWN);
        }
        catch (...) {
            PYNVVC_THROW_ERROR("PyNvMuxer constructor failed with unknown error", CUDA_ERROR_UNKNOWN);
        }
    }

    bool Mux(const std::shared_ptr<PacketData>& packet)
    {
        if (!muxer || !packet) {
            return false;
        }
        return muxer->Mux(*packet);
    }

    bool MuxBitstream(py::buffer bitstream, bool isKeyFrame)
    {
        if (!muxer) {
            return false;
        }

        py::buffer_info info = bitstream.request();
        if (info.ndim != 1 || info.size <= 0) {
            PYNVVC_THROW_ERROR("MuxBitstream expects a 1D non-empty buffer", CUDA_ERROR_NOT_SUPPORTED);
        }

        auto* data = static_cast<uint8_t*>(info.ptr);
        unsigned int size = static_cast<unsigned int>(info.size);

        int64_t pts = nextPts;
        int64_t dts = nextPts;
        int64_t duration = 1;

        bool key = firstPacket ? true : isKeyFrame;
        bool ok = muxer->MuxRaw(data, size, pts, dts, duration, key);
        if (ok) {
            nextPts += 1;
            firstPacket = false;
        }
        return ok;
    }

    void Close()
    {
        muxer.reset();
    }
};

void Init_PyNvMuxer(py::module& m)
{
    m.def(
        "CreateMuxer",
        [](const std::string& filename, std::shared_ptr<PyNvDemuxer> demuxer) {
            return std::make_shared<PyNvMuxer>(filename, std::move(demuxer));
        },
        py::arg("filename"),
        py::arg("demuxer"),
        R"pbdoc(
        Create a muxer bound to an existing PyNvDemuxer.

        The muxer will create a new container at the given filename and
        accepts PacketData instances produced by the demuxer.

        :param filename: Output container path (e.g., .mp4, .mov, .webm).
        :param demuxer:  PyNvDemuxer instance that provides PacketData packets.
    )pbdoc");

    m.def(
        "CreateMuxer",
        [](const std::string& filename,
           const std::string& codec,
           int width,
           int height,
           double fps) {
            return std::make_shared<PyNvMuxer>(filename, codec, width, height, fps);
        },
        py::arg("filename"),
        py::arg("codec"),
        py::arg("width"),
        py::arg("height"),
        py::arg("fps") = 30.0,
        R"pbdoc(
        Create a muxer for encoding workflows, without an existing demuxer.

        The muxer will create a new container at the given filename and
        accepts raw encoded bitstream packets via MuxBitstream().

        :param filename: Output container path (e.g., .mp4, .mkv, .webm).
        :param codec:    Codec name ('h264', 'hevc', or 'av1').
        :param width:    Frame width in pixels.
        :param height:   Frame height in pixels.
        :param fps:      Nominal frame rate used for timestamps.
    )pbdoc");

    py::class_<PyNvMuxer, std::shared_ptr<PyNvMuxer>>(m, "PyNvMuxer", py::module_local())
        .def(
            py::init<const std::string&, std::shared_ptr<PyNvDemuxer>>(),
            py::arg("filename"),
            py::arg("demuxer"),
            R"pbdoc(
        Constructor method. Initialize muxer session with a given
        output container and an associated demuxer providing packets.
    )pbdoc")
        .def(
            py::init<const std::string&, const std::string&, int, int, double>(),
            py::arg("filename"),
            py::arg("codec"),
            py::arg("width"),
            py::arg("height"),
            py::arg("fps") = 30.0,
            R"pbdoc(
        Constructor method. Initialize muxer session for raw encoded
        bitstreams without an existing container/demuxer.

        Use MuxBitstream() to append encoded packets.
    )pbdoc")
        .def(
            "Mux",
            [](std::shared_ptr<PyNvMuxer> self, std::shared_ptr<PacketData> packet) {
                return self->Mux(packet);
            },
            py::arg("packet"),
            R"pbdoc(
        Mux a single PacketData instance into the output container.

        :param packet: PacketData returned by the associated PyNvDemuxer.
        :return: True on success, False otherwise.
    )pbdoc")
        .def(
            "MuxBitstream",
            [](std::shared_ptr<PyNvMuxer> self, py::buffer bitstream, bool is_key_frame) {
                return self->MuxBitstream(bitstream, is_key_frame);
            },
            py::arg("bitstream"),
            py::arg("is_key_frame") = false,
            R"pbdoc(
        Mux a raw encoded bitstream packet into the output container.

        Timestamps are generated sequentially (0, 1, 2, ...) using the
        FPS provided at construction time. The first packet is always
        treated as a key frame; subsequent packets can be marked via
        the is_key_frame flag.

        :param bitstream: Bytes-like object containing encoded video.
        :param is_key_frame: Whether this packet is a key frame.
        :return: True on success, False otherwise.
    )pbdoc")
        .def(
            "Close",
            [](std::shared_ptr<PyNvMuxer> self) {
                self->Close();
            },
            R"pbdoc(
        Explicitly finalize the muxer and close the output container.
        After calling this method, further Mux calls will be no-ops.
    )pbdoc");
}
