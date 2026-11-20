"""Download sqlglot at the pinned commit and unpack it to argv[1] (default /src/sqlglot)."""
import io, sys, tarfile, urllib.request
SHA = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
dest = sys.argv[1] if len(sys.argv) > 1 else "/src/sqlglot"
url = f"https://codeload.github.com/tobymao/sqlglot/tar.gz/{SHA}"
data = urllib.request.urlopen(url, timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data)) as tf:
    for m in tf.getmembers():
        parts = m.name.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        m.name = parts[1]
        if m.issym() or m.islnk():
            continue
        tf.extract(m, dest, filter="data")
print("sqlglot", SHA, "->", dest, file=sys.stderr)
