const WebSocket = require('ws');
const socket = new WebSocket(`ws://localhost:5100`);

socket.addEventListener('open', () => {
    console.log('health.js connected to webSocket server.');
});

exports.getHealth = (req, res, next) => {
    var datetime = new Date();
    var formattedDate = datetime.toLocaleString(); // Format the date to a readable string
    res.send('Lede MV/LV frontend version: ' + process.env.VERSION + ', Date: ' + formattedDate);
};