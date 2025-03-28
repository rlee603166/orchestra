Chart.defaults.color = "#b3b3b3";
Chart.defaults.borderColor = "rgba(255, 255, 255, 0.1)";

const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false,
        },
    },
    elements: {
        line: {
            tension: 0.3,
        },
        point: {
            radius: 0,
            hitRadius: 10,
            hoverRadius: 4,
        },
    },
};

// Loss chart
const lossCtx = document.getElementById("lossChart").getContext("2d");
const lossChart = new Chart(lossCtx, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            {
                label: "Loss",
                data: [],
                borderColor: "#ff6b6b",
                backgroundColor: "rgba(255, 107, 107, 0.1)",
                borderWidth: 2,
                fill: true,
            },
        ],
    },
    options: {
        ...commonOptions,
        scales: {
            x: {
                grid: {
                    color: "rgba(255, 255, 255, 0.05)",
                },
                title: {
                    display: true,
                    text: "Step",
                    color: "#b3b3b3",
                },
            },
            y: {
                grid: {
                    color: "rgba(255, 255, 255, 0.05)",
                },
                title: {
                    display: true,
                    text: "Loss",
                    color: "#b3b3b3",
                },
            },
        },
    },
});

// Accuracy chart
const accCtx = document.getElementById("accuracyChart").getContext("2d");
const accChart = new Chart(accCtx, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            {
                label: "Accuracy",
                data: [],
                borderColor: "#4ecca3",
                backgroundColor: "rgba(78, 204, 163, 0.1)",
                borderWidth: 2,
                fill: true,
            },
        ],
    },
    options: {
        ...commonOptions,
        scales: {
            x: {
                grid: {
                    color: "rgba(255, 255, 255, 0.05)",
                },
                title: {
                    display: true,
                    text: "Step",
                    color: "#b3b3b3",
                },
            },
            y: {
                grid: {
                    color: "rgba(255, 255, 255, 0.05)",
                },
                title: {
                    display: true,
                    text: "Accuracy",
                    color: "#b3b3b3",
                },
                max: 1,
            },
        },
    },
});

let updateInterval;
let isTraining = false;

const mainNodeURL = "http://128.151.20.130:8081";

function updateChart() {
    console.log("fetching");
    fetch(`${mainNodeURL}/training_data`)
        .then(response => response.json())
        .then(data => {
            console.log(JSON.stringify(data, null, 2));
            // Update loss chart
            lossChart.data.labels = data.step;
            lossChart.data.datasets[0].data = data.loss;
            lossChart.update();

            // Update accuracy chart
            accChart.data.labels = data.step;
            accChart.data.datasets[0].data = data.accuracy;
            accChart.update();

            // Update current values
            document.getElementById("currentLoss").textContent =
                data.loss[data.loss.length - 1].toFixed(2);
            document.getElementById("currentAcc").textContent =
                (data.accuracy[data.accuracy.length - 1] * 100).toFixed(0) + "%";
        });
}

function startUpdate() {
    updateInterval = setInterval(updateChart, 10000);
}

function stopUpdate() {
    clearInterval(updateInterval);
}


function startTraining() {
    if (isTraining) return;

    fetch(`${mainNodeURL}/train`, {
        method: "POST",
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Training started:", data);

            isTraining = true;
            document.getElementById("startTraining").textContent = "Training...";
            document.getElementById("startTraining").disabled = true;
            document.getElementById("stopTraining").disabled = false;

            updateChart();
            startUpdate();
        })
        .catch(error => {
            console.error("Error starting training:", error);
            alert("Failed to start training. Please check the console for details.");
        });
}

function stopTraining() {
    if (!isTraining) return;

    fetch(`${mainNodeURL}/stop`, {
        method: "POST",
    })
        .then(response => {

            return response.json();
        })
        .then(data => {
            if (data.status !== "success") {
                throw new Error(`HTTP error! Status: ${data.status}`);
            }
            console.log("Training stopped:", data);

            isTraining = false;
            document.getElementById("startTraining").textContent = "Start Training";
            document.getElementById("stopTraining").disabled = true;
            document.getElementById("startTraining").disabled = false;

            stopUpdate();
        })
        .catch(error => {
            console.error("Error stopping training:", error);
            alert("Failed to stop training. Please check the console for details.");
        });
}


const startButton = document.getElementById("startTraining");
startButton.addEventListener("click", startTraining);

const stopButton = document.getElementById("stopTraining");
stopButton.addEventListener("click", stopTraining);

updateChart();
