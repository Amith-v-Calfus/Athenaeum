FROM golang:1.26-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./
RUN CGO_ENABLED=0 go build -o /gateway .

FROM alpine:3.20

RUN adduser -D -u 10001 gateway
WORKDIR /app
COPY --from=builder /gateway /app/gateway

# The uploads dir is a named volume shared with the worker container; Docker
# initializes a volume's ownership from whatever's at the mount point in the
# image at creation time, so this makes a *fresh* volume writable by the
# non-root gateway user. Existing volumes created before this fix still need
# a one-time `chown` on their current contents.
RUN mkdir -p /data/uploads && chown gateway:gateway /data/uploads

USER gateway
EXPOSE 8080

ENTRYPOINT ["/app/gateway"]
