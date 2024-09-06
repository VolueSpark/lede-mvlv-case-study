exports.postUpdate = (req, res, next) => {
    const data = req.body; // Get the JSON payload from the request body
    console.log('Received JSON data:', data);

    // Example response
    res.status(200).send({
        message: 'Data received successfully',
        receivedData: data
    });
};