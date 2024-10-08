const fs = require('fs');
const path = require('path');
const jsonFilePath = path.join(__dirname, '../public/assets/lede.geojson');

const WebSocket = require('ws');
const socket = new WebSocket(`ws://localhost:5100`);

socket.addEventListener('open', () => {
    console.log('update.js connected to webSocket server.');
});

// Function to send messages
function requestUpdate(timestamp) {
    const message = 'update';
    socket.send(message);
    console.log(`frontend update requested for lfa evaluated at ${timestamp}`)

}

exports.postUpdate = (req, res, next) => {
    const data = req.body; // Get the JSON payload from the request body
    const timestamp = req.query.timestamp;
    console.log(`LFA result at ${timestamp}`);

    // Read the existing GeoJSON file
    fs.readFile(jsonFilePath, 'utf8', (err, fileData) => {
        if (err) {
            console.error('Error reading file:', err);
            return res.status(500).send({ message: 'Error reading file local asset file for updating' });
        }

        // Parse the existing GeoJSON data
        let geoJsonData;
        try {
            geoJsonData = JSON.parse(fileData);
        } catch (parseError) {
            console.error('Error parsing JSON:', parseError);
            return res.status(500).send({ message: 'Error parsing JSON' });
        }

        // Prepare a mapping of id to color from the received data
        const idToColorMap = {};
        data.id.forEach((id, index) => {
            idToColorMap[id] = data.color[index];
        });

        const idToValueMap = {};
        data.id.forEach((id, index) => {
            idToValueMap[id] = data.value[index];
        });

        // Update the color of each feature based on the id
        geoJsonData.features.forEach(feature => {
            const id = feature.properties.id;
            if (idToColorMap[id]) {
                feature.properties.color = idToColorMap[id];
            }
            if (idToValueMap[id]) {
                feature.properties.value = idToValueMap[id];
            }
        });

        // Write the updated data back to the file
        fs.writeFile(jsonFilePath, JSON.stringify(geoJsonData, null, 2), 'utf8', (writeError) => {
            if (writeError) {
                console.error('Error writing file:', writeError);
                return res.status(500).send({ message: 'Error writing file' });
            }

            requestUpdate(timestamp)
            // Send a successful response
            res.status(200).send({
                message: `Lede frontend received data for LFA result at ${timestamp}`,
            });
        });
    });
};