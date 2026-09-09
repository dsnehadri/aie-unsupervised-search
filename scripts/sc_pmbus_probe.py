# System Controller I2C/PMBus probe. Read-only: an address probe plus three
# PMBus word reads. Needs nothing but a python3 with array/struct/fcntl -- the
# SC image has no i2c-tools and no ctypes, so the ioctl structs are built by
# hand and the buffer addresses come from array.buffer_info().
import array, fcntl, os, glob, struct
RDWR = 0x0707
def io(fd, addr, w, nread):
    wb = array.array('B', w if w else [0])
    rb = array.array('B', [0] * max(nread, 1))
    m = struct.pack('<HHH2xQ', addr, 0, len(w), wb.buffer_info()[0])
    if nread:
        m += struct.pack('<HHH2xQ', addr, 1, nread, rb.buffer_info()[0])
    ms = array.array('B'); ms.frombytes(m)
    fcntl.ioctl(fd, RDWR, struct.pack('<QI4x', ms.buffer_info()[0], 2 if nread else 1))
    return list(rb[:nread])
def l11(w):                      # PMBus LINEAR11
    m = w & 0x7FF; e = w >> 11
    if m > 1023: m -= 2048
    if e > 15: e -= 32
    return m * (2.0 ** e)
for dev in sorted(glob.glob("/dev/i2c-*"), key=lambda s: int(s.split("-")[1])):
    try: fd = os.open(dev, os.O_RDWR)
    except OSError: continue
    hits = []
    for a in range(0x08, 0x78):
        try: io(fd, a, [], 1); hits.append(a)
        except OSError: pass
    if hits: print(dev, "->", " ".join("0x%02x" % a for a in hits))
    for a in hits:
        if a == 0x16 or 0x40 <= a <= 0x4f:
            try:
                v = io(fd, a, [0x8B], 2); i = io(fd, a, [0x8C], 2); p = io(fd, a, [0x96], 2)
                print("   0x%02x  VOUT %.3f V  IOUT %.2f A  POUT %.2f W" %
                      (a, l11(v[0] | v[1] << 8), l11(i[0] | i[1] << 8), l11(p[0] | p[1] << 8)))
            except OSError as e:
                print("   0x%02x  no PMBus (%s)" % (a, e.strerror))
    os.close(fd)
