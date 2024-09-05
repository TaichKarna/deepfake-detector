import React, { useState } from 'react';

function VideoUploader() {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        if (!selectedFile) {
            alert('Please select a video file.');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('http://localhost:8000/api/v1/mlapp/upload',  {
                method: "POST",
                body: formData
            });

            console.log('Video uploaded successfully:', response.data);
        } catch (error) {
            console.error('Error uploading video:', error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button type="submit">Upload Video</button>
        </form>
    );
}

export default VideoUploader;
