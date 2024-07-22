import * as tf from '@tensorflow/tfjs';
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
const model = tf.sequential();
model.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [1]}));
model.add(tf.layers.dense({units: 1}));

model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError'
});

model.fit(xs, ys, {epochs: 200}).then(() => {
    drawChart();
});

function makePrediction() {
    const inputValue = document.getElementById('inputValue').value;
    const prediction = model.predict(tf.tensor2d([parseFloat(inputValue)], [1, 1]));
    prediction.array().then(preds => {
        document.getElementById('predictionResult').innerText = `Predicted value: ${preds[0][0].toFixed(2)}`;
    });
}

async function drawChart() {
    const predictions = model.predict(tf.tensor2d([1, 2, 3, 4, 5, 6], [6, 1]));
    const preds = await predictions.array();

    const ctx = document.getElementById('myChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: [1, 2, 3, 4, 5, 6],
            datasets: [{
                label: 'Predicted values',
                data: preds.map(p => p[0]),
                borderColor: 'blue',
                fill: false
            }, {
                label: 'Original values',
                data: [1, 3, 5, 7],
                borderColor: 'red',
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
