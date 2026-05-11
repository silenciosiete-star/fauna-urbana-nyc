window.addEventListener('load', function () {
    // Desbloquea el elemento <audio> en el primer click del usuario para
    // que autoplay funcione cuando el panel reproduce hitos automáticamente.
    // Se usa un WAV silencioso como src porque play() en un elemento sin src
    // falla y no llega a desbloquear el elemento.
    var _WAV_SILENCIO = 'data:audio/wav;base64,UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA';

    function desbloquear() {
        var audio = document.getElementById('audio-tts');
        if (!audio) return;
        if (!audio._desbloqueado) {
            var src_original = audio.src;
            audio.src = _WAV_SILENCIO;
            var p = audio.play();
            if (p !== undefined) {
                p.then(function () {
                    audio.pause();
                    audio.currentTime = 0;
                    audio.src = src_original;
                    audio._desbloqueado = true;
                }).catch(function () {
                    audio.src = src_original;
                });
            }
        }
        document.removeEventListener('click', desbloquear);
    }
    document.addEventListener('click', desbloquear);
});
