<template>
  <div>
    <input v-model="query" placeholder="enter query" />
    <p>Query is: {{ query }}</p>
  </div>
</template>

<script>
  export default {
    data() {
      return {
        query: ''
      }
    }
  }
</script>