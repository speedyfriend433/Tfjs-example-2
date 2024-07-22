import * as tf from '@tensorflow/tfjs';
let canvas, ctx;

async function loadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = async function(event) {
        try {
            const rawData = event.target.result;
            const data = rawData.trim().split('\n').map(row => row.split(',').map(Number));

            if (data.length === 0 || data[0].length < 2) {
                alert('Invalid data format.');
                return;
            }

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

            drawGraph(xs, ys, preds);
        } catch (error) {
            console.error('Error processing file:', error);
            alert('An error occurred while processing the file.');
        }
    };

    reader.readAsText(file);
}

function drawGraph(xs, ys, preds) {
    canvas = document.getElementById('myCanvas');
    ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    const xScale = (width - 2 * padding) / xs.length;
    const yMax = Math.max(...ys.flat(), ...preds.flat());
    const yMin = Math.min(...ys.flat(), ...preds.flat());
    const yScale = (height - 2 * padding) / (yMax - yMin);

    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.lineTo(width - padding, padding);
    ctx.stroke();

    ctx.strokeStyle = 'red';
    ctx.beginPath();
    ctx.moveTo(padding, height - padding - (ys[0][0] - yMin) * yScale);
    for (let i = 1; i < ys.length; i++) {
        ctx.lineTo(padding + i * xScale, height - padding - (ys[i][0] - yMin) * yScale);
    }
    ctx.stroke();

    ctx.strokeStyle = 'blue';
    ctx.beginPath();
    ctx.moveTo(padding, height - padding - (preds[0][0] - yMin) * yScale);
    for (let i = 1; i < preds.length; i++) {
        ctx.lineTo(padding + i * xScale, height - padding - (preds[i][0] - yMin) * yScale);
    }
    ctx.stroke();
}
