<template>
  <div>
    <input v-model="query" placeholder="enter query" />
    <p>Query is: {{ query }}</p>
    <button @click="recieve">beautiful button</button>
  </div>
</template>

<script>
  export default {
    data() {
      return {
        query: ''
      }
    },
    methods: {
      async recieve() {
        const response = await fetch("http://localhost:5000/testGet");
        const data = await response.json();
        console.log(data)
        // try {
        //   fetch("http://localhost:5000/testGet")
        //   .then(response => response.json())
        //   .then(data => (console.log(data)));
        // } catch (e) {
        //   console.log(e)
        // }
      }
    }
  }
</script>
