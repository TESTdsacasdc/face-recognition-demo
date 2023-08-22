async function setupWebcam() {
    const video = document.getElementById('video');

    // Try to get user media
    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function detectFace() {
    // Load the models needed
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
    await faceapi.nets.ageGenderNet.loadFromUri('/models');
    await faceapi.nets.faceExpressionNet.loadFromUri('/models');

    const video = await setupWebcam();
    video.play();

    video.addEventListener('play', () => {
        const canvas = faceapi.createCanvasFromMedia(video);
        document.body.append(canvas);
        const displaySize = { width: video.width, height: video.height };
        faceapi.matchDimensions(canvas, displaySize);

        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                .withFaceLandmarks().withFaceDescriptors().withAgeAndGender().withFaceExpressions();

            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
            faceapi.draw.drawDetections(canvas, resizedDetections);
            faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

            // Convert detections to desired JSON format
            const results = detections.map(d => ({
                faceId: d._imageDims._width + "-" + d._imageDims._height, // This is just a placeholder ID
                faceRectangle: {
                    top: d._box._y,
                    left: d._box._x,
                    width: d._box._width,
                    height: d._box._height
                },
                faceAttributes: {
                    gender: d.gender,
                    age: d.age,
                    emotion: d.expressions
                }
            }));
            console.log(JSON.stringify(results));
        }, 100);
    });
}

detectFace();
