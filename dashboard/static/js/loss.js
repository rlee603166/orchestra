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

function updateChart() {
    console.log("fetching");
    fetch("http://localhost:5000/training_data")
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

updateChart();
setInterval(updateChart, 1000);
