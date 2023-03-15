<template>
  <div class="search-container">
    <div class="search-input">
      <input class="search-bar" v-model="query" placeholder="Search for something" />
    </div>
    <div class="search-button">
      <button class="button-primary" @click="fetch(query)">Search</button>
    </div>
  </div>

  <div class="search-query">
    <p>You are searching for: {{ query }}</p>
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
      async fetch(query) {
        console.log(query);
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

<style>
.search-container {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
  background-color: #fff;
  border-radius: 5px;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
}

.search-input {
  flex: 1;
  margin-right: 10px;
}

.search-bar {
  width: 100%;
  border: none;
  font-size: 16px;
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
}

.search-button {
  margin-left: 10px;
}

.search-button button {
  font-size: 16px;
  font-weight: 500;
  color: #fff;
  background-color: #3b82f6;
  border-radius: 5px;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.search-button button:hover {
  background-color: #2563eb;
}

.search-query {
  margin-top: 10px;
  font-size: 14px;
  color: #555;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}
</style>