"""Minimal single-stream Ogg/Opus muxer & demuxer for the Moshi protocol.

Ported verbatim from the research client (moshi_inference.py). Moshi exchanges
Ogg-framed Opus packets over the WebSocket; this builds the ID/Tags/audio
pages on send and reassembles Opus packets on receive.
"""

from __future__ import annotations

import os
import struct

OGG_CAPTURE = b"OggS"
_CRC_TABLE = None


def _crc_table():
    global _CRC_TABLE
    if _CRC_TABLE is not None:
        return _CRC_TABLE
    table = [0] * 256
    for i in range(256):
        r = i << 24
        for _ in range(8):
            if r & 0x80000000:
                r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                r = (r << 1) & 0xFFFFFFFF
        table[i] = r
    _CRC_TABLE = table
    return table


def _crc32_ogg(data: bytes) -> int:
    table = _crc_table()
    crc = 0
    for b in data:
        idx = ((crc >> 24) ^ b) & 0xFF
        crc = ((crc << 8) ^ table[idx]) & 0xFFFFFFFF
    return crc


class OggOpusMuxer:
    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.serial = int.from_bytes(os.urandom(4), "little")
        self.seq = 0
        self.granule_pos = 0

    def _build_page(self, packet: bytes, header_type: int, granule_pos: int) -> bytes:
        laces = []
        remaining = len(packet)
        while remaining >= 255:
            laces.append(255)
            remaining -= 255
        laces.append(remaining)
        laces_bytes = bytes(laces)

        header = bytearray()
        header += OGG_CAPTURE
        header += struct.pack("<B", 0)
        header += struct.pack("<B", header_type)
        header += struct.pack("<Q", granule_pos)
        header += struct.pack("<I", self.serial)
        header += struct.pack("<I", self.seq)
        header += struct.pack("<I", 0)  # CRC placeholder
        header += struct.pack("<B", len(laces))
        header += laces_bytes

        self.seq += 1
        page = bytearray(bytes(header) + packet)
        page_with_zero_crc = bytearray(page)
        page_with_zero_crc[22:26] = b"\x00\x00\x00\x00"
        crc = _crc32_ogg(bytes(page_with_zero_crc))
        struct.pack_into("<I", page, 22, crc)
        return bytes(page)

    def build_id_header(self) -> bytes:
        head = b"OpusHead"
        head += struct.pack("<B", 1)
        head += struct.pack("<B", self.channels)
        head += struct.pack("<H", 312)
        head += struct.pack("<I", 48000)
        head += struct.pack("<H", 0)
        head += struct.pack("<B", 0)
        return self._build_page(head, 0x02, 0)

    def build_tags_header(self) -> bytes:
        vendor = b"MoshiInferencePy"
        tags = b"OpusTags"
        tags += struct.pack("<I", len(vendor)) + vendor
        tags += struct.pack("<I", 0)
        return self._build_page(tags, 0x00, 0)

    def build_audio_page(self, opus_packet: bytes, samples_per_packet: int = 960) -> bytes:
        self.granule_pos += samples_per_packet
        return self._build_page(opus_packet, 0x00, self.granule_pos)


class OggDemuxer:
    def __init__(self):
        self.buf = b""
        self.packets = []

    def push(self, data: bytes):
        self.buf += data
        while len(self.buf) >= 27:
            if self.buf[:4] != OGG_CAPTURE:
                idx = self.buf.find(OGG_CAPTURE)
                if idx < 0:
                    self.buf = b""
                    return
                self.buf = self.buf[idx:]
                if len(self.buf) < 27:
                    return
            page_segments = self.buf[26]
            if len(self.buf) < 27 + page_segments:
                return
            segment_table = self.buf[27:27 + page_segments]
            body_len = sum(segment_table)
            page_len = 27 + page_segments + body_len
            if len(self.buf) < page_len:
                return
            body = self.buf[27 + page_segments:page_len]
            offset = 0
            partial = b""
            for i in range(page_segments):
                seg = segment_table[i]
                partial += body[offset:offset + seg]
                offset += seg
                if seg < 255:
                    if len(partial) >= 8 and partial[:8] in (b"OpusHead", b"OpusTags"):
                        partial = b""
                        continue
                    self.packets.append(partial)
                    partial = b""
            self.buf = self.buf[page_len:]

    def get_packets(self):
        pkts = self.packets
        self.packets = []
        return pkts
