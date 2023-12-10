        const waveformContainer = document.getElementById('waveform')
        const audioInput = document.getElementById('audio-input')
        const playButton = document.getElementById('play-button')
        const pauseButton = document.getElementById('pause-button')
        const browseButton = document.getElementById('browse-button');
        const downloadButton = document.getElementById('download-button');

        const wavesurfer = WaveSurfer.create({
            container: waveformContainer,
            waveColor: '#00ccff',
            progressColor: '#0088cc'
        });

        const dropArea = document.getElementById('drop-area');

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
            if (!files || !files[0]) {
                return;
            }
            const fileURL = URL.createObjectURL(files[0])
            wavesurfer.load(fileURL)
        }

        playButton.addEventListener('click', function () {
            wavesurfer.play()
        });

        pauseButton.addEventListener('click', function () {
            wavesurfer.pause()
        });

        browseButton.addEventListener('click', function () {
            audioInput.click();
        });
        downloadButton.addEventListener('click', function () {
          // Assuming wavesurfer.load('./assets/ThisOldGuitar.mid') is used initially
          const midiFileURL = './assets/ThisOldGuitar.mid';
          const downloadLink = document.createElement('a');
          downloadLink.href = midiFileURL;
          downloadLink.download = 'downloaded.mid';
          document.body.appendChild(downloadLink);
          downloadLink.click();
          document.body.removeChild(downloadLink);
      });