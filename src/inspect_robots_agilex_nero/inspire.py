"""Inspire RH56 dexterous hand over RS485, using Inspire's own frame protocol.

Source: Inspire Robots, *The Dexterous Hand RH56 Series User Manual*,
PRJ-02-TS-U-001 V1.0.9, section 2.2 (frame format, worked examples) and the
register table. RS485 is 8N1 at 115200 baud by default.

Instruction frames start ``0xEB 0x90``, responses ``0x90 0xEB``. Then come the
hand id, the frame data length, the command (``0x11`` read, ``0x12`` write),
the register byte address (low byte, high byte), the data, and a checksum:
the low byte of the sum of every byte after the two header bytes. Register
values are little-endian 16-bit, six per block, in the DoF order little,
ring, middle, index, thumb bend, thumb rotation. 1000 is open, 0 is closed.

Not yet run on hardware. The manual is the only source for the protocol;
the left/right hand ids on the lent unit are unknown (Unitree's integration
uses right 1, left 2).
"""

from __future__ import annotations

import struct
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

READ = 0x11
WRITE = 0x12
ANGLE_SET = 1486
FORCE_SET = 1498
SPEED_SET = 1522
ANGLE_ACT = 1546
ERROR = 1606
DOF = 6


class SerialPort(Protocol):
    """The slice of ``serial.Serial`` used here."""

    def write(self, data: bytes) -> int | None:
        """Send bytes."""
        ...

    def read(self, size: int) -> bytes:
        """Read up to ``size`` bytes (fewer on timeout)."""
        ...

    def reset_input_buffer(self) -> None:
        """Drop unread input."""
        ...

    def close(self) -> None:
        """Close the port."""
        ...


def checksum(body: bytes) -> int:
    """Low byte of the sum of every byte after the two header bytes."""
    return sum(body) & 0xFF


def read_frame(hand_id: int, address: int, length: int) -> bytes:
    """Instruction to read ``length`` bytes starting at register byte ``address``."""
    body = bytes((hand_id, 0x04, READ, address & 0xFF, address >> 8, length))
    return b"\xeb\x90" + body + bytes((checksum(body),))


def write_frame(hand_id: int, address: int, data: bytes) -> bytes:
    """Instruction to write ``data`` starting at register byte ``address``."""
    body = bytes((hand_id, len(data) + 3, WRITE, address & 0xFF, address >> 8)) + data
    return b"\xeb\x90" + body + bytes((checksum(body),))


def pack_values(values: Sequence[int]) -> bytes:
    """Six DoF values as little-endian 16-bit words (``-1`` means leave as is)."""
    if len(values) != DOF:
        raise ValueError(f"need {DOF} values; got {len(values)}")
    for value in values:
        if value != -1 and not 0 <= value <= 1000:
            raise ValueError(f"Inspire values are 0 to 1000 or -1; got {value}")
    return struct.pack("<6h", *values)


def parse_response(frame: bytes, hand_id: int, command: int, address: int) -> bytes:
    """Check a response frame and return its data bytes."""
    if len(frame) < 9 or frame[:2] != b"\x90\xeb":
        raise RuntimeError(f"Inspire hand: bad response header {frame[:2].hex() or 'none'}")
    body, check = frame[2:-1], frame[-1]
    if checksum(body) != check:
        raise RuntimeError("Inspire hand: response checksum mismatch")
    if body[0] != hand_id or body[2] != command:
        raise RuntimeError(
            f"Inspire hand: response for id {body[0]} command 0x{body[2]:02x}, "
            f"expected id {hand_id} command 0x{command:02x}"
        )
    if body[3] | (body[4] << 8) != address or body[1] != len(body) - 2:
        raise RuntimeError("Inspire hand: response address or length mismatch")
    return bytes(body[5:])


def grip_to_angles(
    grip: float, open_angles: Sequence[int], closed_angles: Sequence[int]
) -> list[int]:
    """Interpolate the six DoF between open (grip 0) and closed (grip 1)."""
    alpha = float(np.clip(grip, 0.0, 1.0))
    opened = np.asarray(open_angles, dtype=np.float64)
    closed = np.asarray(closed_angles, dtype=np.float64)
    return [round(v) for v in opened + alpha * (closed - opened)]


def angles_to_grip(
    angles: Sequence[int], open_angles: Sequence[int], closed_angles: Sequence[int]
) -> float:
    """Project measured angles onto the open-to-closed line; 0 open, 1 closed."""
    opened = np.asarray(open_angles, dtype=np.float64)
    span = np.asarray(closed_angles, dtype=np.float64) - opened
    moving = span != 0
    if not np.any(moving):
        return 0.0
    fraction = (np.asarray(angles, dtype=np.float64) - opened)[moving] / span[moving]
    return float(np.clip(np.mean(fraction), 0.0, 1.0))


class InspireHand:
    """One RH56 hand on a serial port. Every exchange is one request, one response."""

    def __init__(
        self,
        port: SerialPort,
        *,
        hand_id: int,
        open_angles: Sequence[int],
        closed_angles: Sequence[int],
    ) -> None:
        self._port = port
        self._id = hand_id
        self._open = tuple(open_angles)
        self._closed = tuple(closed_angles)
        self._lock = threading.Lock()

    def _exchange(self, request: bytes, command: int, address: int, data_len: int) -> bytes:
        with self._lock:
            self._port.reset_input_buffer()
            self._port.write(request)
            frame = self._port.read(8 + data_len)
        data = parse_response(frame, self._id, command, address)
        if len(data) != data_len:
            raise RuntimeError(f"Inspire hand: expected {data_len} data bytes, got {len(data)}")
        return data

    def write_block(self, address: int, values: Sequence[int]) -> None:
        """Write six DoF values to a register block and check the acknowledgement."""
        data = self._exchange(
            write_frame(self._id, address, pack_values(values)), WRITE, address, 1
        )
        if data != b"\x01":
            raise RuntimeError(f"Inspire hand rejected the write to {address}: {data.hex()}")

    def read_block(self, address: int) -> list[int]:
        """Read a six-value register block."""
        data = self._exchange(read_frame(self._id, address, 2 * DOF), READ, address, 2 * DOF)
        return list(struct.unpack("<6h", data))

    def configure(self, speed: int, force: int) -> None:
        """Set every DoF's speed and force limit."""
        self.write_block(SPEED_SET, [speed] * DOF)
        self.write_block(FORCE_SET, [force] * DOF)

    def set_grip(self, grip: float) -> None:
        """Command the fingers to grip ``grip`` (0 open, 1 closed)."""
        self.write_block(ANGLE_SET, grip_to_angles(grip, self._open, self._closed))

    def read_grip(self) -> float:
        """Measured grip, 0 open to 1 closed."""
        return angles_to_grip(self.read_block(ANGLE_ACT), self._open, self._closed)

    def close(self) -> None:
        """Close the serial port. The fingers hold their last command."""
        self._port.close()


def open_inspire(  # pragma: no cover - hardware
    port: str, baud: int, hand_id: int, open_angles: Sequence[int], closed_angles: Sequence[int]
) -> InspireHand:
    """Open the RS485 port with pyserial."""
    try:
        import serial
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'The Inspire hand needs pyserial: pip install "inspect-robots-agilex-nero[hardware]"'
        ) from exc
    link = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0.1)
    return InspireHand(link, hand_id=hand_id, open_angles=open_angles, closed_angles=closed_angles)
