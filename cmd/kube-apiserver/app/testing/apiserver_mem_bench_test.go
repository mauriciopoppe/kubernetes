/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Reproduction commands (from the root of a kubernetes/kubernetes checkout):
//
//   curl -sL <RAW_GIST_URL> -o cmd/kube-apiserver/app/testing/apiserver_mem_bench_test.go
//   RUN_APISERVER_MEM_BENCH=1 BENCH_OUT_DIR=/tmp/bench-after \
//     go test -v -count=5 -timeout=10m ./cmd/kube-apiserver/app/testing -run=^TestAPIServerMemoryIntegration$
//   go tool pprof -text -alloc_space -base=/tmp/bench-before/apiserver_heap.pb.gz /tmp/bench-after/apiserver_heap.pb.gz

package testing

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"runtime/trace"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type MemoryTelemetry struct {
	HeapInuseMB         float64            `json:"heap_inuse_mb"`
	HeapAllocMB         float64            `json:"heap_alloc_mb"`
	HeapSysMB           float64            `json:"heap_sys_mb"`
	HeapIdleMB          float64            `json:"heap_idle_mb"`
	HeapReleasedMB      float64            `json:"heap_released_mb"`
	SpanFragOverheadMB  float64            `json:"span_frag_overhead_mb"`
	TotalAllocMB        float64            `json:"total_alloc_mb"`
	WorkloadAllocMB     float64            `json:"workload_alloc_mb"`
	Mallocs             uint64             `json:"mallocs"`
	Frees               uint64             `json:"frees"`
	LiveObjects         uint64             `json:"live_objects"`
	NumGC               uint32             `json:"num_gc"`
	GCPauseTotalMs      float64            `json:"gc_pause_total_ms"`
	Goroutines          int                `json:"goroutines"`
	SysMB               float64            `json:"sys_mb"`
	ProcessRSSMB        float64            `json:"process_rss_mb"`
	CgroupWorkingSetMB  float64            `json:"cgroup_working_set_mb"`
	APILatencyMeanMs    float64            `json:"api_latency_mean_ms"`
	EndpointLatenciesMs map[string]float64 `json:"endpoint_latencies_ms"`
	GOGC                int                `json:"gogc"`
}

func readProcessRSSMB() float64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, err := strconv.ParseFloat(fields[1], 64)
				if err == nil {
					return kb / 1024.0
				}
			}
		}
	}
	return 0
}

func readCgroupWorkingSetMB() float64 {
	curBytes, err := os.ReadFile("/sys/fs/cgroup/memory.current")
	if err != nil {
		return 0
	}
	cur, err := strconv.ParseFloat(strings.TrimSpace(string(curBytes)), 64)
	if err != nil {
		return 0
	}
	var inactiveFile float64
	if statBytes, err := os.ReadFile("/sys/fs/cgroup/memory.stat"); err == nil {
		for _, line := range strings.Split(string(statBytes), "\n") {
			if strings.HasPrefix(line, "inactive_file ") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					inactiveFile, _ = strconv.ParseFloat(fields[1], 64)
				}
			}
		}
	}
	ws := cur - inactiveFile
	if ws < 0 {
		ws = cur
	}
	return ws / (1024.0 * 1024.0)
}

// TestAPIServerMemoryIntegration boots an in-process kube-apiserver with embedded etcd,
// seeds 10 bound CEL ValidatingAdmissionPolicies + 60 system ConfigMaps (20 KB each,
// ~1.25 MB per LIST), drives concurrent 8-worker load across ConfigMap Protobuf LIST,
// JSON gzip LIST, GET, UPDATE (triggering CEL admission), and Kubernetes 1.27+ OpenAPI V3
// GroupVersion chunk discovery, runs a single synchronous GC cycle (collecting dead
// request objects while preserving warm sync.Pool victim caches as in steady-state
// production), dumps a live pprof heap profile, and records exact memory telemetry.
func TestAPIServerMemoryIntegration(t *testing.T) {
	if os.Getenv("RUN_APISERVER_MEM_BENCH") != "1" {
		t.Skip("Skipping kube-apiserver memory integration benchmark unless RUN_APISERVER_MEM_BENCH=1")
	}

	gogc := 50
	if v := os.Getenv("GOGC_PERCENTAGE"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			gogc = parsed
		}
	}
	debug.SetGCPercent(gogc)
	runtime.MemProfileRate = 1

	outDir := os.Getenv("BENCH_OUT_DIR")
	if outDir == "" {
		outDir = t.TempDir()
	}
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}

	tCtx := ktesting.Init(t).WithoutCancel()
	_, storageConfig := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

	instanceOptions := NewDefaultTestServerOptions()
	instanceOptions.DisableInvariantChecks = true

	if pPath, err := pkgPath(t); err == nil && pPath != "" {
		_ = os.MkdirAll(filepath.Join(pPath, "testdata"), 0755)
	}

	server, err := StartTestServer(tCtx, instanceOptions, nil, storageConfig)
	if err != nil {
		t.Fatalf("failed to start in-process kube-apiserver: %v", err)
	}
	defer server.TearDownFn()

	jsonConfig := restclient.CopyConfig(server.ClientConfig)
	jsonConfig.QPS = -1
	jsonConfig.Burst = 1000
	client, err := kubernetes.NewForConfig(jsonConfig)
	if err != nil {
		t.Fatalf("failed to create kubernetes client: %v", err)
	}

	protoConfig := restclient.CopyConfig(server.ClientConfig)
	protoConfig.AcceptContentTypes = "application/vnd.kubernetes.protobuf,application/json"
	protoConfig.ContentType = "application/vnd.kubernetes.protobuf"
	protoConfig.QPS = -1
	protoConfig.Burst = 1000
	protoClient, err := kubernetes.NewForConfig(protoConfig)
	if err != nil {
		t.Fatalf("failed to create protobuf client: %v", err)
	}

	transport, err := restclient.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to build transport: %v", err)
	}
	httpClient := &http.Client{Transport: transport, Timeout: 30 * time.Second}
	baseURL := server.ClientConfig.Host

	ctx := context.Background()
	endpointLatencies := make(map[string]float64)

	// 1. Seed 10 ValidatingAdmissionPolicies + 10 ValidatingAdmissionPolicyBindings
	// so CEL environment compilation, regexp compilation, and per-write CEL admission evaluation run.
	failPolicy := admissionregistrationv1.Fail
	matchPolicy := admissionregistrationv1.Equivalent
	for i := 0; i < 10; i++ {
		policyName := fmt.Sprintf("bench-cel-policy-%02d", i)
		vap := &admissionregistrationv1.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: policyName,
			},
			Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
				FailurePolicy: &failPolicy,
				MatchConstraints: &admissionregistrationv1.MatchResources{
					MatchPolicy: &matchPolicy,
					ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
						{
							RuleWithOperations: admissionregistrationv1.RuleWithOperations{
								Operations: []admissionregistrationv1.OperationType{
									admissionregistrationv1.Create,
									admissionregistrationv1.Update,
								},
								Rule: admissionregistrationv1.Rule{
									APIGroups:   []string{""},
									APIVersions: []string{"v1"},
									Resources:   []string{"configmaps"},
								},
							},
						},
					},
				},
				Validations: []admissionregistrationv1.Validation{
					{
						Expression: fmt.Sprintf("has(object.metadata.name) && !object.metadata.name.startsWith('forbidden-%d-') && object.metadata.name.matches('^[a-z0-9-]+$')", i),
						Message:    "ConfigMap name must match allowed pattern and not start with forbidden prefix",
					},
				},
			},
		}
		_, _ = client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(ctx, vap, metav1.CreateOptions{})

		binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("bench-cel-binding-%02d", i),
			},
			Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        policyName,
				ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
			},
		}
		_, _ = client.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
	}

	// 2. Seed 60 representative system ConfigMaps (20 KB each -> ~1.25 MB per LIST response)
	// with labels, annotations, and managedFields.
	const numConfigMaps = 60
	payload := strings.Repeat("system-configuration-payload-line-0123456789abcdef\n", 370)
	for i := 0; i < numConfigMaps; i++ {
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("system-cm-%02d", i),
				Namespace: "kube-system",
				Labels: map[string]string{
					"app.kubernetes.io/managed-by": "kube-apiserver-bench",
					"app.kubernetes.io/component":  "control-plane",
					"k8s-app":                      "kube-apiserver",
					"tier":                         "control-plane",
				},
				Annotations: map[string]string{
					"bench.k8s.io/layer": "core",
				},
			},
			Data: map[string]string{
				"config.yaml": payload,
			},
		}
		if _, err := client.CoreV1().ConfigMaps("kube-system").Create(ctx, cm, metav1.CreateOptions{
			FieldManager: "apiserver-bench-manager",
		}); err != nil {
			t.Fatalf("failed to seed configmap %d: %v", i, err)
		}
	}

	var msAfterBoot runtime.MemStats
	runtime.ReadMemStats(&msAfterBoot)

	// Start CPU profile and Go runtime execution trace across the active concurrent request phase
	cpuProfFile, err := os.Create(filepath.Join(outDir, "apiserver_cpu.pb.gz"))
	if err == nil {
		_ = pprof.StartCPUProfile(cpuProfFile)
	}
	traceFile, err := os.Create(filepath.Join(outDir, "apiserver_trace.out"))
	if err == nil {
		_ = trace.Start(traceFile)
	}

	// 3. Drive concurrent multi-worker load (8 workers x 8 rounds = 64 rounds):
	//    - Protobuf ConfigMap LIST (~1.25 MB per response)
	//    - JSON gzip ConfigMap LIST (~1.25 MB JSON)
	//    - ConfigMap GET (Protobuf + JSON)
	//    - ConfigMap UPDATE (exercises bound CEL ValidatingAdmissionPolicies & ObjectMeta managedFields)
	const numWorkers = 8
	const roundsPerWorker = 8
	var cmLatencyNs atomic.Int64
	var cmErr atomic.Value

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		workerID := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < roundsPerWorker; r++ {
				rt0 := time.Now()
				if _, err := protoClient.CoreV1().ConfigMaps("kube-system").List(ctx, metav1.ListOptions{}); err != nil {
					cmErr.Store(fmt.Errorf("worker %d protobuf list: %w", workerID, err))
					return
				}
				// JSON LIST with gzip compression
				req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/v1/namespaces/kube-system/configmaps", nil)
				if err != nil {
					cmErr.Store(err)
					return
				}
				req.Header.Set("Accept", "application/json")
				req.Header.Set("Accept-Encoding", "gzip")
				resp, err := httpClient.Do(req)
				if err != nil {
					cmErr.Store(fmt.Errorf("worker %d json gzip list: %w", workerID, err))
					return
				}
				if resp.Header.Get("Content-Encoding") == "gzip" && resp.StatusCode == http.StatusOK {
					gr, gerr := gzip.NewReader(resp.Body)
					if gerr == nil {
						_, _ = io.Copy(io.Discard, gr)
						_ = gr.Close()
					}
				} else {
					_, _ = io.Copy(io.Discard, resp.Body)
				}
				_ = resp.Body.Close()

				// 3 Protobuf GETs + 3 JSON GETs
				for i := 0; i < 3; i++ {
					cmIdx := (workerID*6 + r + i) % numConfigMaps
					name := fmt.Sprintf("system-cm-%02d", cmIdx)
					if _, err := protoClient.CoreV1().ConfigMaps("kube-system").Get(ctx, name, metav1.GetOptions{}); err != nil {
						cmErr.Store(fmt.Errorf("worker %d proto get: %w", workerID, err))
						return
					}
					if _, err := client.CoreV1().ConfigMaps("kube-system").Get(ctx, name, metav1.GetOptions{}); err != nil {
						cmErr.Store(fmt.Errorf("worker %d json get: %w", workerID, err))
						return
					}
				}

				// 1 ConfigMap Update to exercise bound CEL ValidatingAdmissionPolicies & ManagedFields
				targetName := fmt.Sprintf("system-cm-%02d", workerID)
				cmObj, err := client.CoreV1().ConfigMaps("kube-system").Get(ctx, targetName, metav1.GetOptions{})
				if err != nil {
					cmErr.Store(fmt.Errorf("worker %d get for update: %w", workerID, err))
					return
				}
				if cmObj.Annotations == nil {
					cmObj.Annotations = make(map[string]string)
				}
				cmObj.Annotations["bench.k8s.io/round"] = strconv.Itoa(r)
				if _, err := client.CoreV1().ConfigMaps("kube-system").Update(ctx, cmObj, metav1.UpdateOptions{
					FieldManager: fmt.Sprintf("bench-worker-%02d", workerID),
				}); err != nil {
					cmErr.Store(fmt.Errorf("worker %d update: %w", workerID, err))
					return
				}
				cmLatencyNs.Add(time.Since(rt0).Nanoseconds())
			}
		}()
	}
	wg.Wait()
	if v := cmErr.Load(); v != nil {
		t.Fatalf("concurrent configmap workload failed: %v", v)
	}
	totalCMRounds := float64(numWorkers * roundsPerWorker)
	endpointLatencies["configmap_list_get_ms"] = (float64(cmLatencyNs.Load()) / 1e6) / totalCMRounds

	// 4. Exercise Kubernetes 1.27+ OpenAPI V3 GroupVersion chunk requests concurrently across 8 workers.
	v3Paths := []string{
		"/openapi/v3/api/v1",
		"/openapi/v3/apis/apps/v1",
		"/openapi/v3/apis/admissionregistration.k8s.io/v1",
	}
	const v3RoundsPerWorker = 2
	var v3LatencyNs atomic.Int64
	var v3Err atomic.Value
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < v3RoundsPerWorker; r++ {
				for _, p := range v3Paths {
					reqT0 := time.Now()
					req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+p, nil)
					if err != nil {
						v3Err.Store(err)
						return
					}
					req.Header.Set("Accept", "application/json")
					req.Header.Set("Accept-Encoding", "gzip")
					resp, err := httpClient.Do(req)
					if err != nil {
						v3Err.Store(fmt.Errorf("GET %s failed: %w", p, err))
						return
					}
					if resp.Header.Get("Content-Encoding") == "gzip" && resp.StatusCode == http.StatusOK {
						gr, gerr := gzip.NewReader(resp.Body)
						if gerr == nil {
							_, _ = io.Copy(io.Discard, gr)
							_ = gr.Close()
						}
					} else {
						_, _ = io.Copy(io.Discard, resp.Body)
					}
					_ = resp.Body.Close()
					v3LatencyNs.Add(time.Since(reqT0).Nanoseconds())
				}
			}
		}()
	}
	wg.Wait()
	if v := v3Err.Load(); v != nil {
		t.Fatalf("concurrent openapi v3 workload failed: %v", v)
	}
	totalV3Reqs := float64(numWorkers * v3RoundsPerWorker * len(v3Paths))
	endpointLatencies["openapi_v3_ms"] = (float64(v3LatencyNs.Load()) / 1e6) / totalV3Reqs

	if traceFile != nil {
		trace.Stop()
		_ = traceFile.Close()
	}
	if cpuProfFile != nil {
		pprof.StopCPUProfile()
		_ = cpuProfFile.Close()
	}

	// 5. Run a single synchronous GC + OS scavenger cycle to collect unreachable
	// request allocations while preserving warm sync.Pool caches.
	runtime.GC()
	debug.FreeOSMemory()
	time.Sleep(50 * time.Millisecond)

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	var profBuf bytes.Buffer
	if err := pprof.WriteHeapProfile(&profBuf); err != nil {
		t.Fatalf("WriteHeapProfile: %v", err)
	}
	profPath := filepath.Join(outDir, "apiserver_heap.pb.gz")
	if err := os.WriteFile(profPath, profBuf.Bytes(), 0644); err != nil {
		t.Fatalf("write %s: %v", profPath, err)
	}

	if goroutineProf := pprof.Lookup("goroutine"); goroutineProf != nil {
		var gBuf bytes.Buffer
		_ = goroutineProf.WriteTo(&gBuf, 0)
		_ = os.WriteFile(filepath.Join(outDir, "apiserver_goroutine.pb.gz"), gBuf.Bytes(), 0644)
	}

	rssMB := readProcessRSSMB()
	wsMB := readCgroupWorkingSetMB()
	if wsMB <= 0 {
		wsMB = rssMB
	}

	var totalLat float64
	for _, v := range endpointLatencies {
		totalLat += v
	}
	meanLat := totalLat / float64(len(endpointLatencies))

	heapInuseMB := float64(ms.HeapInuse) / (1024.0 * 1024.0)
	heapAllocMB := float64(ms.HeapAlloc) / (1024.0 * 1024.0)
	totalAllocMB := float64(ms.TotalAlloc) / (1024.0 * 1024.0)
	workloadAllocMB := float64(ms.TotalAlloc-msAfterBoot.TotalAlloc) / (1024.0 * 1024.0)

	telemetry := MemoryTelemetry{
		HeapInuseMB:         heapInuseMB,
		HeapAllocMB:         heapAllocMB,
		HeapSysMB:           float64(ms.HeapSys) / (1024.0 * 1024.0),
		HeapIdleMB:          float64(ms.HeapIdle) / (1024.0 * 1024.0),
		HeapReleasedMB:      float64(ms.HeapReleased) / (1024.0 * 1024.0),
		SpanFragOverheadMB:  heapInuseMB - heapAllocMB,
		TotalAllocMB:        totalAllocMB,
		WorkloadAllocMB:     workloadAllocMB,
		Mallocs:             ms.Mallocs,
		Frees:               ms.Frees,
		LiveObjects:         ms.Mallocs - ms.Frees,
		NumGC:               ms.NumGC,
		GCPauseTotalMs:      float64(ms.PauseTotalNs) / 1e6,
		Goroutines:          runtime.NumGoroutine(),
		SysMB:               float64(ms.Sys) / (1024.0 * 1024.0),
		ProcessRSSMB:        rssMB,
		CgroupWorkingSetMB:  wsMB,
		APILatencyMeanMs:    meanLat,
		EndpointLatenciesMs: endpointLatencies,
		GOGC:                gogc,
	}

	telemetryBytes, err := json.MarshalIndent(telemetry, "", "  ")
	if err != nil {
		t.Fatalf("marshal telemetry: %v", err)
	}
	telemetryPath := filepath.Join(outDir, "apiserver_telemetry.json")
	if err := os.WriteFile(telemetryPath, telemetryBytes, 0644); err != nil {
		t.Fatalf("write %s: %v", telemetryPath, err)
	}
	t.Logf("APISERVER_MEM_TELEMETRY=%s", string(telemetryBytes))
}
