const ctx = document.getElementById("chart").getContext("2d");
const lossChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            { label: "Loss", data: [], borderColor: "red", yAxisID: "y1" },
            { label: "Accuracy", data: [], borderColor: "blue", yAxisID: "y2" },
        ],
    },
    options: {
        scales: {
            x: { title: { display: true, text: "Step" } },
            y1: { position: "left", title: { display: true, text: "Loss" } },
            y2: { position: "right", title: { display: true, text: "Accuracy" }, max: 1 },
        },
    },
});


let updateInterval;
let isTraining = false;

function updateChart() {
    console.log("fetching");
    fetch('http://47.144.148.193:5000/training_data')
        .then(response => response.json())
        .then(data => {
            lossChart.data.labels = data.steps;
            lossChart.data.datasets[0].data = data.loss;
            // lossChart.data.datasets[1].data = data.accuracy;
            lossChart.update();
            document.getElementById("currentLoss").textContent =
                data.loss[data.loss.length - 1].toFixed(2);
            // document.getElementById("currentAcc").textContent =
            //     (data.accuracy[data.accuracy.length - 1] * 100).toFixed(0) + "%";
        });
}

function startTraining() {
    if (isTraining) return;
    
    fetch('http://47.144.148.193:5000/train', {
        method: 'POST',
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Training started:', data);
            
            isTraining = true;
            document.getElementById('startTraining').textContent = 'Training...';
            document.getElementById('startTraining').disabled = true;
            
            updateChart();
            updateInterval = setInterval(updateChart, 1000);
        })
        .catch(error => {
            console.error('Error starting training:', error);
            alert('Failed to start training. Please check the console for details.');
        });
}

document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startTraining');
    startButton.addEventListener('click', startTraining);
});

updateChart();
