<template>
  <div>
    <div class="search-container">
      <div class="search-input">
        <img src="./assets/GoogIR.png" width="150"/>
        <input class="search-bar" v-model="query" v-on:keyup.enter="fetch(query)" placeholder="Search for something" />
        <div class="search-button">
          <button class="button-primary" @click="fetch(query)">Search</button>
        </div>
      </div>
    </div>
    <div class="search-query" v-if="timeTaken !== -1">
      <p>Time elapsed: {{ timeTaken }} seconds</p>
    </div>
    <LoadingScreen v-if="isLoading" />
    <div class="search-result" v-else v-for="item in data">
      <a :href="item[1]" target="_blank">
        <h3 class="search-result-title" @click='window.location.href = item[1]'>{{ item[2] }}</h3>
      </a>
      <p class="search-result-abstract">{{ getLimitedAbstract(item[6]) }}</p>
    </div>
  </div>
</template>

<script>
import LoadingScreen from "./LoadingScreen";

export default {
  data() {
    return {
      query: '',
      maxAbstractWords: 50,
      timeTaken: -1,
      isLoading: false,
      // Data index meanings
      // 0: uuid, 1: repository link, 2: title, 3: author, 4: contributor, 5: publication year, 6: abstract, 7: subject topic,
      // 8: language, 9: publication type, 10: publisher, 11: isbn, 12: issn, 13: patent, 14: patent status, 15: bibliographic note,
      // 16: access restriction, 17: embargo date, 18: faculty, 19: department, 20: research group, 21: programme, 22: project, 23: coordinates
      data: '',
      
    }
  },
  methods: {
    async fetch(query) {
      this.isLoading = true;
      const options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'query': query })
      };
      
      fetch('http://localhost:5000/retrieve', options)
        .then(async response => {
          const data = await response.json();
          if (!response.ok) {
            const error = (data && data.message) || response.status;
            return Promise.reject(error);
          }
          this.timeTaken = data.time;
          this.data = data.results;
          this.isLoading = false;
        })
        .catch(error => {
          this.errorMessage = error;
          this.isLoading = false;
          console.error('There was an error!', error);
        });
    }
  },
  computed: {
    getLimitedAbstract() {
      return function(abstract) {
        const words = abstract.split(" ");
        if (words.length > this.maxAbstractWords) {
          return words.slice(0, this.maxAbstractWords).join(" ") + " ...";
        } else {
          return abstract;
        }
      };
    }
  },
  components: {
    LoadingScreen
  }
}
</script>

<style>
.search-container {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
  background-color: #fff;
  border-radius: 5px;
}

.search-input {
  flex: 1;
  display: flex;
  align-items: center;
}

.search-bar {
  width: 80%;
  margin-left: 25px;
  /* margin-top: -100px; */
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
  color: #808080;
  font-size: 12px;
  max-width: 1000px;
  margin: 0 auto;
  padding-left: 20px;
  padding-right: 20px;
  background-color: #fff;
  border-radius: 5px;
}

.search-result {
  max-width: 1000px;
  margin: 0 auto;
  padding: 10px;
  background-color: #fff;
  border-radius: 5px;
}

.search-result-title {
  font-size: 26px;
  font-weight: bold;
}

.search-result-abstract {
  font-size: 1.2rem;
  color: #808080;
  margin-top: 0.5rem;
}
</style>
