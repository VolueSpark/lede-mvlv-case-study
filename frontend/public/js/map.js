mapboxgl.accessToken =  window.MAPBOX_TOKEN;//'pk.eyJ1IjoicGhpbGxpcG1hcmVlIiwiYSI6ImNseXI5aDk3ODA0ZjkycnMxcHg5NmNqMzIifQ.qdYicgPirCFPEj2q2ime8w';
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [9.694324395243175, 59.12449186974682], // starting position [lng, lat]. Note that lat must be set between -90 and 90
    zoom: 14.2 // starting zoom
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
        data: '../assets/lede.base.geojson'
    });

    map.loadImage('../icons/home.png', (error, image) => {
        if (error) throw error;
        if (!map.hasImage('home')) map.addImage('home', image, { 'sdf': true });
    });

    map.loadImage('../icons/substation.png', (error, image) => {
        if (error) throw error;
        if (!map.hasImage('substation')) map.addImage('substation', image, { 'sdf': true });
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
            'line-width': 1
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
            'icon-allow-overlap': false, // Allow icons to overlap
            'text-field': ['get', 'name'], // Fetch the text from the 'name' property in the GeoJSON
            'text-size': 12, // Adjust the text size (values like 2 are too small)
            'text-offset': [0, 1.2], // Offset the text above the icon
            'text-anchor': 'top', // Position text above the icon
        },
        "paint": {
            "icon-color": ['get', 'color']
        }
    });

    map.addLayer({
        'id': `power-transformer`,
        'type': 'symbol',
        'source': 'geojson_id',
        'filter': ['==', ['get', 'objecttype'], 'PowerTransformer'],
        'layout': {
            'icon-image': 'substation', // Replace with the Maki icon name from Mapbox
            'icon-size': 1, // Adjust icon size if needed
            'icon-allow-overlap': true, // Allow icons to overlap
            'text-size': 14, // Adjust the text size (values like 2 are too small)
            'text-offset': [0, 1.2], // Offset the text above the icon
            'text-anchor': 'top' // Position text above the icon
        },
        "paint": {
            "icon-color": ['get', 'color'],
            "icon-halo-width": 0
        }
    });

    // Create a popup, but don't add it to the map yet.
    const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false,
        maxWidth: '600px'
    });

    map.on('mouseenter', 'power-transformer', (e) => {
        // Change the cursor style as a UI indicator.
        map.getCanvas().style.cursor = 'pointer';

        // Copy coordinates array.
        const coordinates = e.features[0].geometry.coordinates.slice();
        const description = '<strong>Transformer name:</strong> '+e.features[0].properties.name
            + '<br><strong>Topology ID:</strong> ' + e.features[0].properties.topology_id
            + '<br><strong>Transformer loading:</strong> ' + e.features[0].properties.value;

        // Ensure that if the map is zoomed out such that multiple
        // copies of the feature are visible, the popup appears
        // over the copy being pointed to.
        if (['mercator', 'equirectangular'].includes(map.getProjection().name)) {
            while (Math.abs(e.lngLat.lng - coordinates[0]) > 180) {
                coordinates[0] += e.lngLat.lng > coordinates[0] ? 360 : -360;
            }
        }

        // Populate the popup and set its coordinates
        // based on the feature found.
        popup.setLngLat(coordinates).setHTML(description).addTo(map);
    });

    map.on('mouseenter', 'conform-load', (e) => {
        // Change the cursor style as a UI indicator.
        map.getCanvas().style.cursor = 'pointer';

        // Copy coordinates array.
        const coordinates = e.features[0].geometry.coordinates.slice();
        const description = '<strong>Conform load mrid:</strong> '+e.features[0].properties.cfl_id
            +'<br><strong>Voltage:</strong> ' + e.features[0].properties.value;

        // Ensure that if the map is zoomed out such that multiple
        // copies of the feature are visible, the popup appears
        // over the copy being pointed to.
        if (['mercator', 'equirectangular'].includes(map.getProjection().name)) {
            while (Math.abs(e.lngLat.lng - coordinates[0]) > 180) {
                coordinates[0] += e.lngLat.lng > coordinates[0] ? 360 : -360;
            }
        }

        // Populate the popup and set its coordinates
        // based on the feature found.
        popup.setLngLat(coordinates).setHTML(description).addTo(map);
    });

    map.on('mouseleave', 'conform-load', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
    });

}

map.on('load', loadGrid);



