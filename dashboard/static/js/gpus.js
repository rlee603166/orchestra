document.addEventListener("DOMContentLoaded", function () {
    let gpuData = [];
    const urls = [
        "http://128.151.20.95:5000",
        "http://128.151.20.120:5000",
        "http://128.151.20.140:5000",
        "http://128.151.20.147:5000",
        "http://128.151.20.156:5000",
        "http://128.151.20.197:5000",
        "http://128.151.20.235:5000"
    ];

    function renderGPUNodes() {
        const title = document.getElementById("gpu-header");
        if (title) {
            if (gpuData.length > 0) {
                if (gpuData.length === 1) {
                    title.textContent = `GPU Grid: [${gpuData.length} node]`;
                } else {
                    title.textContent = `GPU Grid: [${gpuData.length} nodes]`;
                }
            } else {
                title.textContent = "Connect a compute node!";
            }
        } else {
            console.error("Element with ID 'gpu-header' not found");
        }
        const nodeList = document.getElementById("nodeList");
        nodeList.innerHTML = "";

        gpuData.forEach(gpu => {
            const memoryPercent = (gpu.memory / gpu.memoryTotal) * 100;

            let utilClass = "util-idle";
            if (gpu.utilization > 0 && gpu.utilization < 75) {
                utilClass = "util-normal";
            } else if (gpu.utilization >= 75 && gpu.utilization < 90) {
                utilClass = "util-high";
            } else if (gpu.utilization >= 90) {
                utilClass = "util-critical";
            }

            let memoryFillClass = "";
            if (memoryPercent >= 85 && memoryPercent < 95) {
                memoryFillClass = "high";
            } else if (memoryPercent >= 95) {
                memoryFillClass = "critical";
            }

            let nodeClass = "gpu-node";
            if (gpu.status === "offline") {
                nodeClass += " offline";
            } else if (gpu.utilization >= 90 || memoryPercent >= 95) {
                nodeClass += " critical";
            } else if (gpu.utilization >= 75 || memoryPercent >= 85) {
                nodeClass += " high-load";
            }

            const nodeElement = document.createElement("div");
            nodeElement.className = nodeClass;

            nodeElement.innerHTML = `
                <div class="gpu-header">
                    <span class="gpu-name">${gpu.id}</span>
                    <span class="status-indicator ${gpu.status}"></span>
                </div>
                <div class="gpu-stats">
                    <div class="stat-row">
                        <span class="stat-label">Util:</span>
                        <span class="stat-value ${utilClass}">${gpu.utilization}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Mem:</span>
                        <span class="stat-value">${gpu.memory}G / ${gpu.memoryTotal}G</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Temp:</span>
                        <span class="stat-value">${gpu.temp}°C</span>
                    </div>
                </div>
                <div class="memory-bar">
                    <div class="memory-fill ${memoryFillClass}" style="width: ${memoryPercent}%"></div>
                </div>
            `;

            nodeList.appendChild(nodeElement);
        });
    }

    function fetchNodeData(url, id) {
        fetch(`${url}/stats`)
            .then(response => response.json())
            .then(data => {
                console.log(data, null, 2);

                const formattedGpu = {
                    id: data.rank,
                    utilization: data.gpu_util,
                    memory: data.gpu_stats.vram_used_gpu,
                    memoryTotal: data.gpu_stats.vram_total_gb,
                    temp: data.gpu_temp,
                    status: data.status || "idle",
                };

                const existingIndex = gpuData.findIndex(gpu => gpu.id === formattedGpu.id);
                if (existingIndex >= 0) {
                    gpuData[existingIndex] = formattedGpu;
                } else {
                    gpuData.push(formattedGpu);
                }

                renderGPUNodes();
                console.log(`Updated gpuData: ${JSON.stringify(gpuData, null, 2)}`);
            })
            .catch(error => {
                console.error("Error fetching GPU data:", error);
            });
    }

    function updateGPUData() {
        urls.forEach((url, index) => {
            fetchNodeData(url, index);
        });
    }

    renderGPUNodes();
    updateGPUData();

    setInterval(updateGPUData, 15000);
});
