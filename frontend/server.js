const path = require('path');
const express = require('express');
const ws = require('ws')
const dotenv = require('dotenv');
const cors = require('cors');
const connectDB = require('./config/db');

// load env var
dotenv.config({path:'./config/config.env'});

// connect to database
connectDB();

const app = express();

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

const PORT = process.env.PORT || 3000
const httpServer = app.listen(PORT, ()=>
    console.log(`Server running in ${process.env.NODE_ENV} mode on port ${PORT}`)
);
const wsServer = new ws.Server({ noServer: true })
httpServer.on('upgrade', (req, socket, head) => {
    wsServer.handleUpgrade(req, socket, head, (ws) => {
        wsServer.emit('connection', ws, req)
    })
})


