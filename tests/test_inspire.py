"""Inspire RH56 frames against the manual's worked examples, and the hand driver."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from inspect_robots_agilex_nero.inspire import (
    ANGLE_ACT,
    ANGLE_SET,
    FORCE_SET,
    SPEED_SET,
    InspireHand,
    angles_to_closure,
    checksum,
    closure_to_angles,
    pack_values,
    parse_response,
    read_frame,
    write_frame,
)

OPEN = (1000,) * 6
CLOSED = (0, 0, 0, 0, 0, 1000)  # thumb rotation fixed open


def test_frames_match_the_manual_examples() -> None:
    # Manual 2.2.1: read ANGLE_ACT (1546, 12 bytes) from hand 1, checksum 0x32.
    assert read_frame(1, ANGLE_ACT, 12) == bytes.fromhex("eb90 0104 110a 060c 32")
    # Manual 2.2.2: write ANGLE_SET 100,100,100,100,1000,0 to hand 1, checksum 0x70.
    data = pack_values([100, 100, 100, 100, 1000, 0])
    assert data == bytes.fromhex("6400 6400 6400 6400 e803 0000")
    assert write_frame(1, ANGLE_SET, data) == bytes.fromhex("eb90010f12ce05") + data + b"\x70"
    # The read response in 2.2.1 parses back to those six values, checksum 0xAC.
    response = bytes.fromhex("90eb010f110a06") + data + b"\xac"
    assert struct.unpack("<6h", parse_response(response, 1, 0x11, ANGLE_ACT)) == (
        100,
        100,
        100,
        100,
        1000,
        0,
    )
    assert checksum(b"\x01\x04\x12\xce\x05\x01") == 0xEB  # write ack example


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (b"", "bad response header"),
        (bytes.fromhex("eb90010412ce050100"), "bad response header"),
        (bytes.fromhex("90eb010412ce050100"), "checksum"),
        (bytes.fromhex("90eb020412ce0501ec"), "id 2"),
        (bytes.fromhex("90eb010412cf0501ec"), "address or length"),
    ],
)
def test_bad_responses_are_rejected(frame: bytes, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        parse_response(frame, 1, 0x12, ANGLE_SET)


def test_values_are_range_checked() -> None:
    assert pack_values([-1] * 6) == b"\xff\xff" * 6
    with pytest.raises(ValueError, match="0 to 1000"):
        pack_values([1001, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="need 6"):
        pack_values([0] * 5)


def test_closure_mapping_is_per_finger_and_round_trips() -> None:
    assert closure_to_angles([0.0] * 6, OPEN, CLOSED) == list(OPEN)
    assert closure_to_angles([0.0, 0.25, 0.5, 1.0, 7.0, 1.0], OPEN, CLOSED) == [
        1000,
        750,
        500,
        0,
        0,
        1000,
    ]
    closure = angles_to_closure([1000, 750, 500, 0, -5, 123], OPEN, CLOSED)
    np.testing.assert_allclose(closure, (0.0, 0.25, 0.5, 1.0, 1.0, 0.0))
    np.testing.assert_allclose(angles_to_closure([1200] * 6, OPEN, CLOSED), 0.0)
    with pytest.raises(ValueError):
        closure_to_angles([0.0] * 5, OPEN, CLOSED)


class FakePort:
    """Answers each request like an RH56 with id 1."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.angles = [1000] * 6
        self.reply_override: bytes | None = None
        self.closed = False

    def reset_input_buffer(self) -> None:
        return None

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def read(self, size: int) -> bytes:
        if self.reply_override is not None:
            return self.reply_override
        request = self.written[-1]
        address = request[5] | (request[6] << 8)
        if request[4] == 0x12:
            if address == ANGLE_SET:
                self.angles = list(struct.unpack("<6h", request[7:-1]))
            body = bytes((1, 4, 0x12, request[5], request[6], 1))
        else:
            data = struct.pack("<6h", *self.angles)
            body = bytes((1, len(data) + 3, 0x11, request[5], request[6])) + data
        frame = b"\x90\xeb" + body + bytes((checksum(body),))
        assert len(frame) == size
        return frame

    def close(self) -> None:
        self.closed = True


def test_hand_driver_configures_grips_and_reads() -> None:
    port = FakePort()
    hand = InspireHand(port, hand_id=1, open_angles=OPEN, closed_angles=CLOSED)
    hand.configure(speed=400, force=300)
    assert [f[5] | (f[6] << 8) for f in port.written] == [SPEED_SET, FORCE_SET]
    assert port.written[0][7:-1] == pack_values([400] * 6)
    hand.set_fingers(np.asarray((1.0, 1.0, 1.0, 0.5, 1.0, 0.0)))
    assert port.angles == [0, 0, 0, 500, 0, 1000]
    np.testing.assert_allclose(hand.read_fingers(), (1.0, 1.0, 1.0, 0.5, 1.0, 0.0))
    hand.close()
    assert port.closed


def test_hand_driver_surfaces_nacks_and_short_replies() -> None:
    port = FakePort()
    hand = InspireHand(port, hand_id=1, open_angles=OPEN, closed_angles=CLOSED)
    nack = bytes((1, 4, 0x12, ANGLE_SET & 0xFF, ANGLE_SET >> 8, 0))
    port.reply_override = b"\x90\xeb" + nack + bytes((checksum(nack),))
    with pytest.raises(RuntimeError, match="rejected"):
        hand.set_fingers(np.full(6, 0.5))
    short = bytes((1, 5, 0x11, ANGLE_ACT & 0xFF, ANGLE_ACT >> 8, 0, 0))
    port.reply_override = b"\x90\xeb" + short + bytes((checksum(short),))
    with pytest.raises(RuntimeError, match="expected 12 data bytes"):
        hand.read_fingers()
