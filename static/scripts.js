document.addEventListener("DOMContentLoaded", function () {
    // Function to handle frame display
    function showFrame(frameId) {
        // Hide all frames
        const frames = document.querySelectorAll('.frame');
        frames.forEach(frame => frame.classList.remove('active'));

        const btn_nav = document.querySelectorAll('.rounded-button');
        btn_nav.forEach(btn => btn.classList.toggle('active'));

        const nav = document.querySelectorAll('.button-container');
        nav.forEach(btn => btn.classList.toggle('nav'));

        // Show the selected frame
        const frameToShow = document.getElementById(frameId);
        if (frameToShow) {
            frameToShow.classList.add('active');
        }
    }

    // Fetch detected objects using GET request
    // function fetchDetectedObjects() {
    //     fetch("/detected_objects", {
    //         method: "GET"
    //     })
    //     .then(response => response.json())
    //     .then(data => {
    //         const detectedList = document.getElementById("detectedObjects");
    //         detectedList.innerHTML = "";  // Clear previous detections

    //         data.forEach(obj => {
    //             let listItem = document.createElement("li");
    //             listItem.textContent = obj.object + " (" + obj.color + ")";
    //             detectedList.appendChild(listItem);
    //         });
    //     });
    // }

    // Make sure the frame functionality works
    window.showFrame = showFrame; // Expose showFrame globally
    // fetchDetectedObjects();  // Load detected objects when the page is ready
});

document.getElementById("speakButton").addEventListener("click", function () {
    fetch("/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
    })
    .then(response => {
        if (response.ok) {
            return response.blob();  // Convert response to blob (audio file)
        } else {
            throw new Error('Failed to fetch audio file');
        }
    })
    .then(blob => {
        console.log("Audio blob received");  // Log for debugging

        const audio = new Audio(URL.createObjectURL(blob));
        audio.play();  // Play the received audio file

        audio.onplay = () => console.log("Audio is playing...");  // Log when audio starts playing
        audio.onerror = (err) => console.error("Audio error:", err);  // Log any audio errors
    })
    .catch(error => {
        console.error("Error:", error);  // Log any errors during the fetch process
    });
}
);
