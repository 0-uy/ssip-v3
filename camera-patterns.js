/**
 * camera-patterns.js - Diccionario de URLs para cámaras IP
 * SSIP v4.0
 */

export const CAMERA_PATTERNS = {
    hikvision: {
        name: 'Hikvision',
        mjpeg: [
            'http://{user}:{pass}@{ip}/Streaming/channels/1/picture',
            'http://{user}:{pass}@{ip}/ISAPI/Streaming/channels/101/picture'
        ]
    },
    dahua: {
        name: 'Dahua',
        mjpeg: [
            'http://{user}:{pass}@{ip}/cgi-bin/mjpg/video.cgi?channel=1',
            'http://{user}:{pass}@{ip}/cgi-bin/snapshot.cgi'
        ]
    },
    axis: {
        name: 'Axis',
        mjpeg: [
            'http://{user}:{pass}@{ip}/axis-cgi/mjpg/video.cgi',
            'http://{user}:{pass}@{ip}/jpg/image.jpg'
        ]
    }
};

console.log('✅ camera-patterns.js cargado');