mapboxgl.accessToken =  window.MAPBOX_TOKEN;//'pk.eyJ1IjoicGhpbGxpcG1hcmVlIiwiYSI6ImNseXI5aDk3ODA0ZjkycnMxcHg5NmNqMzIifQ.qdYicgPirCFPEj2q2ime8w';
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [9.665452, 59.146073], // starting position [lng, lat]. Note that lat must be set between -90 and 90
    zoom: 14 // starting zoom
});

const socket = new WebSocket(`ws://localhost:5100`);

socket.addEventListener('open', () => {
    console.log('map.js connected to webSocket server.');
});

socket.onopen = () =>
{
    console.log('onopen')
}; // When a message is received from the server
socket.onmessage = (message) => {
    console.log('map.js received notification')
    map.getSource('geojson_id').setData('../assets/lede.geojson');
};

// Function Declaration
// icons from https://docs.mapbox.com/data/tilesets/reference/mapbox-streets-v8/
function loadGrid() {

    map.addSource('geojson_id', {
        type: 'geojson',
        data: '../assets/lede.geojson'
    });

    map.loadImage('../icons/home.png', (error, image) => {
        if (error) throw error;
        if (!map.hasImage('home')) map.addImage('home', image, { 'sdf': true });
    });

    map.loadImage('../icons/transformer.png', (error, image) => {
        if (error) throw error;
        if (!map.hasImage('transformer')) map.addImage('transformer', image, { 'sdf': true });
    });

    map.loadImage('../icons/substation.png', (error, image) => {
        if (error) throw error;
        if (!map.hasImage('substation')) map.addImage('substation', image, { 'sdf': true });
    });

    map.addLayer({
        'id': `ac-line-segments`,
        'type': 'line',
        'source': 'geojson_id',
        'filter': ['==', ['get', 'objecttype'], 'AcLineSegment'],
        'layout': {
            'line-join': 'round',
            'line-cap': 'round'
        },
        'paint': {
            'line-color': ['get', 'color'],
            'line-width': 1
        }
    });

    map.addLayer({
        'id': `power-transformer`,
        'type': 'symbol',
        'source': 'geojson_id',
        'filter': ['==', ['get', 'objecttype'], 'PowerTransformer'],
        'layout': {
            'icon-image': 'substation', // Replace with the Maki icon name from Mapbox
            'icon-size': 1.2, // Adjust icon size if needed
            'icon-allow-overlap': false, // Allow icons to overlap
            'text-field': ['get', 'name'], // Fetch the text from the 'name' property in the GeoJSON
            'text-size': 12, // Adjust the text size (values like 2 are too small)
            'text-offset': [0, 1.2], // Offset the text above the icon
            'text-anchor': 'top' // Position text above the icon
        },
        "paint": {
            "icon-color": ['get', 'color'],
            "icon-halo-width": 0
        }
    });

    map.addLayer({
        'id': `conform-load`,
        'type': 'symbol',
        'source': 'geojson_id',
        'filter': ['==', ['get', 'objecttype'], 'ConformLoad'],
        'layout': {
            'icon-image': 'home', // Replace with the Maki icon name from Mapbox
            'icon-size': 0.3, // Adjust icon size if needed
            'icon-allow-overlap': true, // Allow icons to overlap
            //'text-field': ['get', 'cfl_id'], // Fetch the text from the 'name' property in the GeoJSON
            'text-size': 12, // Adjust the text size (values like 2 are too small)
            'text-offset': [0, 1.2], // Offset the text above the icon
            'text-anchor': 'top', // Position text above the icon
        },
        "paint": {
            "icon-color": ['get', 'color']
        }
    });

}

map.on('load', loadGrid);



