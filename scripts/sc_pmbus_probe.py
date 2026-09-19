# System Controller I2C/PMBus probe. Read-only. No i2c-tools and no ctypes on
# that image, so the I2C_RDWR structs are packed by hand and buffer addresses
# come from array.buffer_info(). Reports why a bus is skipped rather than
# staying silent.
import array, fcntl, os, glob, struct
RDWR = 0x0707
def io(fd, addr, w, nread):
    wb = array.array('B', w if w else [0])
    rb = array.array('B', [0] * max(nread, 1))
    parts = []
    if w:     parts.append(struct.pack('<HHH2xQ', addr, 0, len(w), wb.buffer_info()[0]))
    if nread: parts.append(struct.pack('<HHH2xQ', addr, 1, nread, rb.buffer_info()[0]))
    ms = array.array('B'); ms.frombytes(b"".join(parts))
    fcntl.ioctl(fd, RDWR, struct.pack('<QI4x', ms.buffer_info()[0], len(parts)))
    return list(rb[:nread])
def l11(w):
    m = w & 0x7FF; e = w >> 11
    if m > 1023: m -= 2048
    if e > 15: e -= 32
    return m * (2.0 ** e)
devs = sorted(glob.glob("/dev/i2c-*"), key=lambda s: int(s.split("-")[1]))
print("buses found:", len(devs), "uid:", os.getuid())
opened = 0
for dev in devs:
    try:
        fd = os.open(dev, os.O_RDWR)
    except OSError as e:
        print("%s: cannot open (%s)" % (dev, e.strerror)); continue
    opened += 1
    hits = []
    for a in range(0x08, 0x78):
        try: io(fd, a, [], 1); hits.append(a)
        except OSError: pass
    print("%s: %d device(s)%s" % (dev, len(hits),
          ("  " + " ".join("0x%02x" % a for a in hits)) if hits else ""))
    for a in hits:
        if a == 0x16 or 0x40 <= a <= 0x4f:
            try:
                v = io(fd, a, [0x8B], 2); i = io(fd, a, [0x8C], 2); p = io(fd, a, [0x96], 2)
                print("   0x%02x  VOUT %.3f V  IOUT %.2f A  POUT %.2f W" %
                      (a, l11(v[0] | v[1] << 8), l11(i[0] | i[1] << 8), l11(p[0] | p[1] << 8)))
            except OSError as e:
                print("   0x%02x  no PMBus (%s)" % (a, e.strerror))
    os.close(fd)
print("opened", opened, "of", len(devs), "buses")
