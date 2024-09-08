const WebSocket = require('ws');
const wss_port = process.env.WEBSOCKET_PORT
const socket = new WebSocket(`ws://localhost:${wss_port}`);

socket.addEventListener('open', () => {
    console.log('health.js connected to webSocket server.');
});

exports.getHealth = (req, res, next) => {
    var datetime = new Date();
    var formattedDate = datetime.toLocaleString(); // Format the date to a readable string
    res.send('Lede MV/LV frontend version: ' + process.env.VERSION + ', Date: ' + formattedDate);
};