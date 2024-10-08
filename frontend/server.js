const path = require('path');
const express = require('express');
const WebSocket = require('ws');
const dotenv = require('dotenv');
const cors = require('cors');
const connectDB = require('./config/db');

// load env var
dotenv.config({path:'./config/config.env'});

const port = process.env.PORT || 3000
const mode = process.env.NODE_ENV

// connect to database
// connectDB();

const app = express();
const wss = new WebSocket.Server({ port: 5100 });

// WebSocket event handling
wss.on('connection', (ws) => {
    console.log('A new client connected.');

    // Event listener for incoming messages
    ws.on('message', (message) => {
        console.log('Received message to be broadcast to clients:', message.toString());

        // Broadcast the message to all connected clients
        wss.clients.forEach((client) => {
            console.log('Send a new message to client')
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

// Define the CORS options
const corsOptions = {
    credentials: true,
    origin: ['http://localhost:5000', 'ws://localhost:5100'] // Whitelist the domains you want to allow
};

app.use(cors(corsOptions));

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


