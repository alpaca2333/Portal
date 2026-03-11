async function test() {
    try {
        const response = await fetch('http://127.0.0.1:8080/api/quant/search?query=000001');
        const data = await response.json();
        console.log(JSON.stringify(data));
    } catch (e) {
        console.error(e);
    }
}
test();
