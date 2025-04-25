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


let countingInterval = null;
let previousRightFingers = null;
let previousLeftFingers = null;
const startButton = document.getElementById('start_counting');
const stopButton = document.getElementById('stop_counting');

function updateFingerCounts() {
    fetch('/finger_data')
        .then(response => response.json())
        .then(data => {
            const currentRightFingers = data.right;
            const currentLeftFingers = data.left;

            if (currentRightFingers !== previousRightFingers || currentLeftFingers !== previousLeftFingers) {
                document.getElementById('right_finger_count').innerText = currentRightFingers;
                document.getElementById('left_finger_count').innerText = currentLeftFingers;
                previousRightFingers = currentRightFingers;
                previousLeftFingers = currentLeftFingers;
            }
        })
        .catch(error => {
            console.error('Lỗi khi lấy dữ liệu ngón tay:', error);
        });
}

startButton.addEventListener('click', () => {
    if (!countingInterval) {
        updateFingerCounts(); // Gọi ngay khi bật
        countingInterval = setInterval(updateFingerCounts, 200);
        startButton.classList.add('active');
        stopButton.classList.remove('active');
        console.log('Đếm ngón tay đã bật.');
    } else {
        console.log('Đếm ngón tay đã được bật rồi.');
    }
});

stopButton.addEventListener('click', () => {
    if (countingInterval) {
        clearInterval(countingInterval);
        countingInterval = null;
        startButton.classList.remove('active');
        stopButton.classList.add('active');
        document.getElementById('right_finger_count').innerText = 0;
        document.getElementById('left_finger_count').innerText = 0;
        previousRightFingers = null;
        previousLeftFingers = null;
        console.log('Đếm ngón tay đã tắt.');
    } else {
        console.log('Đếm ngón tay chưa được bật.');
    }
});

// Mặc định trạng thái tắt khi trang tải
stopButton.classList.add('active');

