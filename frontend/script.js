let model;

// ===============================
// Firebase configuration
// ===============================
const firebaseConfig = {
  apiKey: "AIzaSyBIunAYGeLgqeTFnUehi6gJM18mqfHFy70",
  authDomain: "cry-detector-673d8.firebaseapp.com",
  databaseURL: "https://cry-detector-673d8-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "cry-detector-673d8",
  storageBucket: "cry-detector-673d8.firebasestorage.app",
  messagingSenderId: "884439639217",
  appId: "1:884439639217:web:c8d268f3ab2933051dee90"
};

firebase.initializeApp(firebaseConfig);
const database = firebase.database();

// ===============================
// Load TensorFlow.js model
// ===============================
async function loadModel() {
    try {
        model = await tf.loadLayersModel('web_model/model.json');
        document.getElementById('model-status').innerText = "Model Loaded ✅";
        console.log("✅ Model loaded. Input shape:", model.inputs[0].shape);
    } catch(err) {
        console.error("Model load failed:", err);
        document.getElementById('model-status').innerText = "Model load failed ❌";
    }
}
loadModel();

// ===============================
// Listen for new audio in Firebase
// ===============================
database.ref('raw_audio').on('child_added', async (snapshot) => {
    const base64Audio = snapshot.val();
    console.log("New audio uploaded:", snapshot.key);

    if(!model) {
        console.log("Model not loaded yet. Skipping audio.");
        return;
    }

    try {
        const inputTensor = await preprocessAudio(base64Audio);

        // Ensure input shape is [1, 40, 174, 1]
        const reshapedInput = inputTensor.reshape([1, 40, 174, 1]);

        const prediction = model.predict(reshapedInput);
        const predictedClass = prediction.argMax(-1).dataSync()[0] === 0 ? "Not Cry" : "Cry";

        // Update UI
        document.getElementById('prediction').innerText = predictedClass;
        const log = document.getElementById('log');
        const li = document.createElement('li');
        li.innerText = `Prediction: ${predictedClass} at ${new Date().toLocaleTimeString()}`;
        log.prepend(li);

        // Push prediction to Firebase
        database.ref('predictions').push({
            value: predictedClass,
            timestamp: Date.now()
        });
    } catch(err) {
        console.error("Error processing audio:", err);
    }
});

// ===============================
// Preprocess audio to a fixed-length tensor
// ===============================
async function preprocessAudio(base64Audio) {
    // Decode Base64 to bytes
    const audioBytes = Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0));

    // Decode WAV with Web Audio API
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(audioBytes.buffer);

    let channelData = audioBuffer.getChannelData(0);

    // Downsample to 16kHz
    const sampleRate = audioBuffer.sampleRate;
    if(sampleRate !== 16000) {
        const factor = sampleRate / 16000;
        const newLength = Math.floor(channelData.length / factor);
        const downsampled = new Float32Array(newLength);
        for(let i=0;i<newLength;i++){
            downsampled[i] = channelData[Math.floor(i*factor)];
        }
        channelData = downsampled;
    }

    // Placeholder MFCC
    const fixedLength = 40*174;
    let mfccData = new Float32Array(fixedLength);
    mfccData.set(channelData.slice(0,fixedLength));
    return tf.tensor(mfccData, [1,40,174,1]);
}
