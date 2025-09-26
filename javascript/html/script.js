// script.js
document.addEventListener("DOMContentLoaded", function() {
    const btn = document.querySelector('.toggle-btn');
    const sidebar = document.querySelector('.sidebar');

    btn.addEventListener('click', function() {
        // Toggle the 'closed' class on the sidebar
        sidebar.classList.toggle('closed')
    });
});