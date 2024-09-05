mapboxgl.accessToken =  window.MAPBOX_TOKEN;//'pk.eyJ1IjoicGhpbGxpcG1hcmVlIiwiYSI6ImNseXI5aDk3ODA0ZjkycnMxcHg5NmNqMzIifQ.qdYicgPirCFPEj2q2ime8w';
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [9.655241966247559, 59.15074157714844], // starting position [lng, lat]. Note that lat must be set between -90 and 90
    zoom: 17 // starting zoom
});

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
            'line-width': 1.5
        }
    });



    map.addLayer({
        'id': `power-transformer`,
        'type': 'symbol',
        'source': 'geojson_id',
        'filter': ['==', ['get', 'objecttype'], 'PowerTransformer'],
        'layout': {
            'icon-image': 'transformer', // Replace with the Maki icon name from Mapbox
            'icon-size': 1, // Adjust icon size if needed
            'icon-allow-overlap': true, // Allow icons to overlap
            'text-field': ['get', 'id'], // Display feature name as label
            'text-size': 1,
            'text-offset': [0, 1.2] // Offset text below the icon
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
            'icon-size': 0.5, // Adjust icon size if needed
            'icon-allow-overlap': true, // Allow icons to overlap
            'text-field': ['get', 'id'], // Display feature name as label
            'text-size': 1,
            'text-offset': [0, 1.2] // Offset text below the icon
        },
        "paint": {
            "icon-color": ['get', 'color']
        }
    });

}

map.on('load', loadGrid);