# --------------------------------------------------
# Stage 1: Build the Rust binary
# --------------------------------------------------

FROM rust:1-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release


# --------------------------------------------------
# Stage 2: Runtime
# --------------------------------------------------

FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY reranker ./reranker
COPY verifier ./verifier

COPY --from=builder /app/target/release/auditgate /usr/local/bin/auditgate

EXPOSE 3000

CMD ["auditgate", "serve"]