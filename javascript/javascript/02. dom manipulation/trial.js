// trial.js

const buttonListener = document.querySelector('.lower-button').addEventListener('click', onClick);

function onClick(e) {
    console.log(`X: ${e.offsetX}\nY: ${e.offsetY}`);
    console.log(e.type);
}