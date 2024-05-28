function saveLetter() {
    fetch('/save_letter', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            const savedLettersDiv = document.getElementById('saved-letters');
            savedLettersDiv.innerHTML = data.saved_letters.join(' ');
        });
}
