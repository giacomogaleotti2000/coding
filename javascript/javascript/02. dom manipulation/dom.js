// dom.js

// now we have to add to each list item the button to remove it
const listItems = document.querySelectorAll('.list-group-item');
listItems.forEach(item => {
    const btn = document.createElement('button');
    btn.classList.add('delete-button');
    btn.textContent = 'Remove';
    item.appendChild(btn);
});

// now the function to remove the item if the button is clicked
listItems.forEach(item => {
    const btn = item.querySelector('.delete-button');
    btn.addEventListener('click', () => {
        item.remove();
    });
});

const item = document.querySelector('.form-control.mr-2')
const btn = document.querySelector('.btn.btn-dark')
const list = document.querySelector('.list-group')

// adding new items to the list --> and adding the event listener to each new button !!
btn.addEventListener('click', (e) => {
    e.preventDefault();

    // add the new item to the list
    const newItem = document.createElement('li');
    newItem.classList.add('list-group-item');
    newItem.textContent = item.value;

    const delBtn = document.createElement('button');
    delBtn.classList.add('delete-button');
    delBtn.textContent = 'Remove';
    newItem.appendChild(delBtn);

    delBtn.addEventListener('click', () => {
        newItem.remove();
    });

    list.appendChild(newItem);
    item.value = '';

});

