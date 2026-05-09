window.addEventListener('load', function () {
    // Desbloquea el elemento <audio> en el primer click del usuario para
    // que autoplay funcione cuando el panel reproduce hitos automáticamente.
    function desbloquear() {
        var audio = document.getElementById('audio-tts');
        if (audio && !audio._desbloqueado) {
            audio._desbloqueado = true;
            var p = audio.play();
            if (p !== undefined) {
                p.then(function () { audio.pause(); audio.currentTime = 0; }).catch(function () {});
            }
        }
        document.removeEventListener('click', desbloquear);
    }
    document.addEventListener('click', desbloquear);
});
