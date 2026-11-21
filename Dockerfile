# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image the commits were labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /app
COPY tools/fetch_sqlglot.py /app/fetch_sqlglot.py
# pristine sqlglot at the pinned commit (the source every patch is applied to)
RUN python /app/fetch_sqlglot.py /app/src && chmod -R a+rX /app/src
# pristine bytecode (build time, not charged): patched files are still compiled from source
RUN python -m compileall -q /app/src/sqlglot /app/src/tests
COPY maps /app/maps
COPY selector.py /app/selector.py
# maps + reference index -> /app/index.pkl (build time, not charged)
RUN python /app/selector.py --prep && chmod -R a+rX /app
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-I", "/app/selector.py"]
