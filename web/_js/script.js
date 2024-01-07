var waveFormContainer;
var audioInput;
var playButton;
var pauseButton;
var resetButton;
var browseButton;
var downloadButton;
var waveSurfer;
var dropArea;
var midiPlayer;
var midiVisualizer;

// when document is ready
$.when( $.ready ).then(function() {
    initialize_waveform();
    initialize_audio_input();
    initialize_buttons();
    initialize_midi();
});

function initialize_buttons() {
    playButton = document.getElementById('play-button')
    pauseButton = document.getElementById('pause-button')
    browseButton = document.getElementById('browse-button');
    resetButton = document.getElementById('reset-button');
    downloadButton = document.getElementById('download-button');

    playButton.addEventListener('click', function () {
        waveSurfer.play()
    });
    
    pauseButton.addEventListener('click', function () {
        waveSurfer.pause()
    });

    resetButton.addEventListener('click', function () {
        // put needle back to start of audio clip
        waveSurfer.stop()
    });

    downloadButton.addEventListener('click', function () {
        // download midi file from URL
        const url = midiPlayer.prop("src");
        const filename = url.split('/').pop();
        download(filename, url); 
    });
}

function download(filename, url) {
    // create invisible link element
    const link = document.createElement('a');
    link.style.display = 'none';
    document.body.appendChild(link);

    // set link href to url
    link.href = url;
    link.download = filename;

    // click link to download file
    link.click();

    // remove link from DOM
    document.body.removeChild(link);

}

function initialize_audio_input() {
    audioInput = document.getElementById('audio-input');
    dropArea = document.getElementById('drop-area');

    dropArea.addEventListener('dragover', function (event) {
        event.preventDefault();
        dropArea.style.backgroundColor = 'rgb(132, 129, 129)';
    });
    
    dropArea.addEventListener('dragleave', function () {
        dropArea.style.backgroundColor = 'rgb(132, 129, 129)';
    });
    
    dropArea.addEventListener('drop', function (event) {
        event.preventDefault();
        dropArea.style.backgroundColor = 'rgb(132, 129, 129)';
    
        const files = event.dataTransfer.files;
        handleFiles(files);
    });
    
    audioInput.addEventListener('change', function () {
        const files = this.files;
        handleFiles(files);
    });
    
    function handleFiles(files) {
        console.log("Handling files...")
        const label = $("#file_label");
        if (!files || !files[0]) {
            label.html("Drag & drop your audio file here <br> or");
            reset_waveform();
            return;
        }
        const fileURL = URL.createObjectURL(files[0])
        waveSurfer.load(fileURL)

        // set label to filename
        const filename = files[0].name;
        label.html("Selected file: " + filename + "<br><br>");
    }
}

function initialize_waveform() {
    waveformContainer = document.getElementById('waveform-container')
    
    waveSurfer = WaveSurfer.create({
        container: waveformContainer,
        waveColor: '#00ccff',
        progressColor: '#0088cc'
    });
}

function reset_waveform() {
    waveSurfer.empty();
}

function initialize_midi() {
    midiPlayer = $("#midi-player");
    midiVisualizer = $("#midi-visualizer");   
}

function load_midi(filepath) {
    midiPlayer.prop("src", filepath);
    midiVisualizer.prop("src", filepath);
}

function reset_midi() {
    midiPlayer.prop("src", "");
    midiVisualizer.prop("src", "");
}