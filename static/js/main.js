let menu = document.querySelector('.menu')
let sidebar = document.querySelector('.sidebar')
let mainContent = document.querySelector('.main--content')

menu.onclick = function() {
    sidebar.classList.toggle('active')
    mainContent.classList.toggle('active')
}

function applyFilter(filterType) {
fetch('/apply_filter', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: 'filter=' + filterType,
});
}

