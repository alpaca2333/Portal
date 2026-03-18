async function test() {
    try {
        const response = await fetch('http://127.0.0.1:8080/api/stock-data/search?query=平安');
        const data = await response.json();
        console.log(JSON.stringify(data));
    } catch (e) {
        console.error(e);
    }
}
test();
