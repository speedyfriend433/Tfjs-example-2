import * as tf from '@tensorflow/tfjs';
let chart;

async function trainAndPredict() {
    const rawData = document.getElementById('inputData').value;
    const data = rawData.trim().split('\n').map(row => row.split(',').map(Number));
    const xs = data.map(row => row.slice(0, -1));
    const ys = data.map(row => row.slice(-1));
    const inputTensor = tf.tensor2d(xs);
    const outputTensor = tf.tensor2d(ys);
    const model = tf.sequential();
    
    model.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [xs[0].length]}));
    model.add(tf.layers.dense({units: 1}));
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });

    await model.fit(inputTensor, outputTensor, {epochs: 200});

    const predictions = model.predict(inputTensor);
    const preds = await predictions.array();

    updateChart(xs, ys, preds);
}

function updateChart(xs, ys, preds) {
    const ctx = document.getElementById('myChart').getContext('2d');

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: xs.map((_, i) => i + 1),
            datasets: [{
                label: 'Actual values',
                data: ys.map(y => y[0]),
                borderColor: 'red',
                fill: false
            }, {
                label: 'Predicted values',
                data: preds.map(p => p[0]),
                borderColor: 'blue',
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });
}
