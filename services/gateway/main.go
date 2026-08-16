package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// IngestionJob mirrors shared/schemas/ingestion_job.schema.json.
// This is the contract the Python worker reads. Keep the json tags in sync
// with that schema file; the schema is the source of truth.
type IngestionJob struct {
	JobID            string `json:"job_id"`
	OriginalFilename string `json:"original_filename"`
	StoragePath      string `json:"storage_path"`
	ContentType      string `json:"content_type"`
	SizeBytes        int64  `json:"size_bytes"`
	UploadedAt       string `json:"uploaded_at"`
	Source           string `json:"source"`
}

// config holds everything the gateway reads from the environment at startup.
type config struct {
	port      string
	redisURL  string
	uploadDir string
	queueName string
	maxBytes  int64
}

// allowedExtensions maps the file extensions the gateway will accept at intake
// to their MIME type. New formats (scanned PDF via OCR, PPTX) plug in here
// later without touching the rest of the gateway.
//
// NOTE: .csv is accepted at intake now so the gateway doesn't need another
// change later, but the Python worker's loader does not yet implement CSV
// parsing (deferred to v2 -- see load_document in tasks.py). A CSV upload
// will be accepted and queued here, then fail loudly and specifically in the
// worker with a clear "not yet implemented" error, rather than being silently
// mishandled as prose text.
var allowedExtensions = map[string]string{
	".pdf":  "application/pdf",
	".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	".html": "text/html",
	".htm":  "text/html",
	".csv":  "text/csv",
}

func loadConfig() config {
	rawUploadDir := getenv("UPLOAD_DIR", "./data/uploads")
	uploadDir, err := filepath.Abs(rawUploadDir)
	if err != nil {
		log.Fatalf("could not resolve upload dir %q to absolute path: %v", rawUploadDir, err)
	}

	return config{
		port:      getenv("PORT", "8080"),
		redisURL:  getenv("REDIS_URL", "redis://localhost:6379/0"),
		uploadDir: uploadDir,
		queueName: getenv("QUEUE_NAME", "ingestion_jobs"),
		maxBytes:  int64(getenvInt("MAX_UPLOAD_SIZE_MB", 50)) * 1024 * 1024,
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil {
			return n
		}
	}
	return fallback
}

// newUUIDv4 generates a random RFC 4122 version-4 UUID using crypto/rand.
// Hand-rolled so the gateway needs no external UUID dependency.
func newUUIDv4() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}

type server struct {
	cfg config
	rdb *redis.Client
}

func main() {
	cfg := loadConfig()

	if err := os.MkdirAll(cfg.uploadDir, 0o755); err != nil {
		log.Fatalf("could not create upload dir %q: %v", cfg.uploadDir, err)
	}

	opts, err := redis.ParseURL(cfg.redisURL)
	if err != nil {
		log.Fatalf("invalid REDIS_URL %q: %v", cfg.redisURL, err)
	}
	rdb := redis.NewClient(opts)

	s := &server{cfg: cfg, rdb: rdb}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/upload", s.handleUpload)

	addr := ":" + cfg.port
	log.Printf("athenaeum gateway listening on %s (upload dir: %s, queue: %s)", addr, cfg.uploadDir, cfg.queueName)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// handleHealth is a liveness check. It also pings Redis so the endpoint fails
// if the queue is unreachable, which is what actually matters for ingestion.
func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.rdb.Ping(ctx).Err(); err != nil {
		http.Error(w, "redis unreachable", http.StatusServiceUnavailable)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleUpload is the whole job of the gateway: receive a file, run cheap
// structural checks, save it to the shared volume, and push a job pointing at
// it. It never opens or parses the file contents; that is the worker's job.
func (s *server) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Cap the request body so an oversized upload cannot exhaust memory/disk.
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.maxBytes)
	if err := r.ParseMultipartForm(s.cfg.maxBytes); err != nil {
		http.Error(w, "file too large or malformed multipart form", http.StatusRequestEntityTooLarge)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing 'file' field in multipart form", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// --- Validation (cheap, structural, no content parsing) ---
	ext := strings.ToLower(filepath.Ext(header.Filename))
	contentType, ok := allowedExtensions[ext]
	if !ok {
		http.Error(w, fmt.Sprintf("unsupported file type %q; allowed: .pdf, .docx, .html, .htm, .csv", ext), http.StatusUnsupportedMediaType)
		return
	}
	if header.Size <= 0 {
		http.Error(w, "empty file", http.StatusBadRequest)
		return
	}

	// --- Save to the shared volume under a generated job id ---
	jobID, err := newUUIDv4()
	if err != nil {
		http.Error(w, "could not generate job id", http.StatusInternalServerError)
		return
	}
	storagePath := filepath.Join(s.cfg.uploadDir, jobID+ext)

	dst, err := os.Create(storagePath)
	if err != nil {
		http.Error(w, "could not save file", http.StatusInternalServerError)
		return
	}
	written, err := io.Copy(dst, file)
	if closeErr := dst.Close(); closeErr != nil && err == nil {
		err = closeErr
	}
	if err != nil {
		// Best-effort cleanup so we don't leave a half-written file behind.
		_ = os.Remove(storagePath)
		http.Error(w, "could not write file", http.StatusInternalServerError)
		return
	}

	// --- Build the job and push it onto the queue ---
	job := IngestionJob{
		JobID:            jobID,
		OriginalFilename: header.Filename,
		StoragePath:      storagePath,
		ContentType:      contentType,
		SizeBytes:        written,
		UploadedAt:       time.Now().UTC().Format(time.RFC3339),
		Source:           "manual-upload",
	}

	payload, err := json.Marshal(job)
	if err != nil {
		_ = os.Remove(storagePath)
		http.Error(w, "could not encode job", http.StatusInternalServerError)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if err := s.rdb.RPush(ctx, s.cfg.queueName, payload).Err(); err != nil {
		// If we cannot enqueue, remove the saved file so nothing is orphaned.
		_ = os.Remove(storagePath)
		http.Error(w, "could not enqueue job", http.StatusServiceUnavailable)
		return
	}

	log.Printf("queued job %s (%s, %d bytes)", jobID, header.Filename, written)
	writeJSON(w, http.StatusAccepted, map[string]string{
		"job_id": jobID,
		"status": "queued",
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
