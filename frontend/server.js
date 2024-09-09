const path = require('path');
const express = require('express');
const WebSocket = require('ws');
const dotenv = require('dotenv');
const cors = require('cors');
const connectDB = require('./config/db');

// load env var
dotenv.config({path:'./config/config.env'});

const port = process.env.PORT || 3000
const wss_port = process.env.WEBSOCKET_PORT
const mode = process.env.NODE_ENV

// connect to database
connectDB();

const app = express();
const wss = new WebSocket.Server({ port: wss_port });

// WebSocket event handling
wss.on('connection', (ws) => {
    console.log('A new client connected.');

    // Event listener for incoming messages
    ws.on('message', (message) => {
        console.log('Received message to be broadcast to clients:', message.toString());

        // Broadcast the message to all connected clients
        wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message.toString());
            }
        });
    });

    // Event listener for client disconnection
    ws.on('close', () => {
        console.log('A client disconnected.');
    });
});


// body parser
app.use(express.json({ limit: '50mb' })); // Set the limit according to your needs
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// enable cors
app.use(cors());

// set static folder
app.use(express.static(path.join(__dirname, 'public')))

// set env variable
app.get('/config.js', (req, res) => {
    res.setHeader('Content-Type', 'application/javascript');
    res.send(`window.MAPBOX_TOKEN = '${process.env.MAPBOX_TOKEN}';`);
});

// routes
app.use('/api/v1/lede', require('./routes/health'))
app.use('/api/v1/lede', require('./routes/update'))

app.listen(port, ()=>
    console.log(`Server running in ${mode} mode on port ${port}`)
);


