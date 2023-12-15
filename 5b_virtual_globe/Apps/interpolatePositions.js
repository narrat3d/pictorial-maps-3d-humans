async function interpolatePositions(Cesium, stopTime, startTime) {
    const data = await Cesium.GeoJsonDataSource.load("geodata/etzel.geojson");

    const positions = data.entities.values[0].polyline.positions.getValue();
    const startPosition = positions[0];
    const endPosition = positions[positions.length - 1];

    const secondsTotal = Cesium.JulianDate.secondsDifference(stopTime, startTime);

    const distances = [];
    let distance;

    //calculate the length of each line segment (neglecting earth curvature)
    //source: https://gis.stackexchange.com/questions/175399/cesium-js-line-length/175430
    for (let i = 1; i < positions.length; i++) {
        distance = Cesium.Cartesian3.distance(positions[i - 1], positions[i]);
        distances.push(distance);
    }

    //calculate the total length of the polyline
    const distanceTotal = distances.reduce((a, b) => {
        return a + b;
    });

    //calculate the velocity which is needed that the 3D model reaches the final position in the given time
    const velocity = distanceTotal / secondsTotal;

    const previousPosition = startPosition;
    let previousTime = startTime;

    const positionProperty = new Cesium.SampledPositionProperty();
    //begin the animation at the start position and time
    positionProperty.addSample(previousTime, previousPosition);

    for (let i = 1; i < positions.length; i++) {
        const position = positions[i];
        const currentDistance = distances[i - 1];
        //calculate the time needed to move along this line segment
        const secondsForLineSegment = currentDistance / velocity;

        previousTime = Cesium.JulianDate.addSeconds(
            previousTime,
            secondsForLineSegment,
            new Cesium.JulianDate()
        );
        positionProperty.addSample(previousTime, position);
    }

    return positionProperty;
}
