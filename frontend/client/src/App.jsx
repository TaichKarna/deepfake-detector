
import '@mantine/core/styles.css';

import { MantineProvider } from '@mantine/core';
import MainRouter from './utils/router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient();

export default function App() {
  
  return (
    <MantineProvider>
      <QueryClientProvider client={queryClient}>
        <MainRouter/>
      </QueryClientProvider>
    </MantineProvider>
  );
}