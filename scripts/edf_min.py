"""Minimal EDF/EDF+ reader (numpy memmap): channel labels, sampling rates, physical-unit signals. No dependencies.
ponytail: reads one channel at a time from int16 records; add pyedflib if EDF+ annotations or BDF (24-bit) are ever needed."""
import numpy as np


def header(path):
    with open(path, "rb") as f:
        h = f.read(256); ns = int(h[252:256]); H = f.read(256 * ns)
    field = lambda off, w: [H[off + i * w: off + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]
    o = 0; out = {}
    for name, w in (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("reserved", 32)):
        out[name] = field(o, w); o += w * ns
    nrec, dur = int(h[236:244]), float(h[244:252])
    out.update(nrec=nrec, dur=dur, hbytes=256 * (ns + 1), ns=ns)
    out["nsamp"] = [int(x) for x in out["nsamp"]]; out["fs"] = [n / dur for n in out["nsamp"]]
    return out


def signal(path, ch, H=None):
    """physical-unit samples of channel index ch"""
    H = H or header(path); per = sum(H["nsamp"]); off = sum(H["nsamp"][:ch]); n = H["nsamp"][ch]
    m = np.memmap(path, np.int16, "r", offset=H["hbytes"], shape=(H["nrec"], per))
    d = np.asarray(m[:, off:off + n], np.float64).ravel()
    pmin, pmax, dmin, dmax = (float(H[k][ch]) for k in ("pmin", "pmax", "dmin", "dmax"))
    return (d - dmin) * (pmax - pmin) / (dmax - dmin) + pmin


if __name__ == "__main__":                                       # self-check on a synthetic 2-channel EDF
    import tempfile, os
    fs, nrec = 4, 3; x = np.arange(nrec * fs, dtype=np.int16)
    def pad(s, w): return s.ljust(w)[:w].encode()
    hd = pad("0", 8) + pad("", 80) + pad("", 80) + pad("01.01.01", 8) + pad("00.00.00", 8) + pad(str(256 * 3), 8) + pad("", 44) + pad(str(nrec), 8) + pad("1", 8) + pad("2", 4)
    cols = [("A", "B"), ("", ""), ("uV", "uV"), ("-100", "-100"), ("100", "100"), ("-100", "-100"), ("100", "100"), ("", ""), (str(fs), str(fs)), ("", "")]
    ws = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32); hd += b"".join(pad(v, w) for c, w in zip(cols, ws) for v in c)
    data = b"".join(np.concatenate([x[r * fs:(r + 1) * fs], -x[r * fs:(r + 1) * fs]]).astype("<i2").tobytes() for r in range(nrec))
    p = os.path.join(tempfile.mkdtemp(), "t.edf"); open(p, "wb").write(hd + data)
    H = header(p); assert H["label"] == ["A", "B"] and H["fs"] == [4.0, 4.0]
    assert np.allclose(signal(p, 0, H), x) and np.allclose(signal(p, 1, H), -x); print("edf_min ok")
